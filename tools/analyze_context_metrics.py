"""Recompute CPU-only context metrics and paired task bootstrap intervals."""
from pathlib import Path
import argparse,json,csv
import numpy as np

NAMES={'sft_full': 'SFT完整历史', 'sft_mask4': 'SFT＋ObsLite', 'sft_window2': 'SFT两步O/A窗口', 'grpo20': 'GRPO20', 'mask4_grpo20': 'ObsLite＋GRPO20'}
PAIRS=[('sft_full', 'sft_mask4'), ('sft_full', 'sft_window2'), ('grpo20', 'mask4_grpo20')]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def stats(v):
 a=np.asarray(v,dtype=float)
 return {'mean':float(a.mean()),'p95':float(np.quantile(a,.95,method='linear')),'max':float(a.max())}
def without_p95_columns(text):
    output=[]
    keep=None
    for line in text.splitlines():
        if line.startswith('|'):
            cells=[v.strip() for v in line.strip().strip('|').split('|')]
            if any('P95' in c for c in cells):
                keep=[i for i,c in enumerate(cells) if 'P95' not in c]
            if keep is not None:
                line='| '+' | '.join(cells[i] for i in keep)+' |'
        else:
            keep=None
        output.append(line)
    return '\n'.join(output)+'\n'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data={p.stem:json.loads(p.read_text(encoding='utf-8-sig')) for p in sorted(a.input.glob('*.json'))} if a.input.is_dir() else json.loads(a.input.read_text(encoding='utf-8-sig'));a.output.mkdir(parents=True,exist_ok=True)
 summary={};curves=[]
 for key,run in data.items():
  if 'missing' in run:continue
  eps=run['episodes'];assert [e['task_id'] for e in eps]==list(range(500)),key
  assert all(e['status']=='completed' and e['steps'] for e in eps)
  calls=[s['prompt_tokens'] for e in eps for s in e['steps']]
  assert all(isinstance(x,int) and x>=0 for x in calls),key
  assert all(isinstance(s['completion_tokens'],int) for e in eps for s in e['steps'])
  durations=[e['elapsed_seconds'] for e in eps]
  timing=run.get('timing.json',{})
  summary[key]={'name':NAMES[key],'tasks':500,'calls':len(calls),'input_per_call':stats(calls),'peak_input_per_episode':stats([max(s['prompt_tokens'] for s in e['steps']) for e in eps]),'episode_input_mean':sum(calls)/500,'episode_output_mean':sum(s['completion_tokens'] for e in eps for s in e['steps'])/500,'end_to_end_seconds':stats(durations) if all(isinstance(v,(int,float)) for v in durations) else None,'evaluation_seconds':timing.get('evaluation_seconds'),'timed_new_episodes':timing.get('new_episodes'),'workers':timing.get('workers'),'success_rate':sum(e['reward']==1 for e in eps)/500,'reward_mean':sum(e['reward'] for e in eps)/500}
  for turn in range(1,max(len(e['steps']) for e in eps)+1):
   v=[e['steps'][turn-1]['prompt_tokens'] for e in eps if len(e['steps'])>=turn]
   curves.append(dict(method=key,turn=turn,remaining_tasks=len(v),**stats(v)))
 write(a.output/'context_summary.json',summary)
 with (a.output/'context_by_turn.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 intervals=[]
 for base,method in PAIRS:
  if base not in summary or method not in summary:continue
  x=np.array([e['reward'] for e in data[base]['episodes']]);y=np.array([e['reward'] for e in data[method]['episodes']])
  for metric,d,scale in [('success_pp',(y==1).astype(float)-(x==1).astype(float),100),('reward',y-x,1)]:
   rng=np.random.default_rng(42);samples=[]
   for _ in range(20):samples.extend(d[rng.integers(0,500,size=(1000,500))].mean(axis=1)*scale)
   lo,hi=np.quantile(samples,[.025,.975]);intervals.append(dict(baseline=base,method=method,metric=metric,difference=float(d.mean()*scale),ci95=[float(lo),float(hi)],bootstrap_samples=20000,seed=42,paired_by='task_id',method_ci='percentile'))
 write(a.output/'paired_confidence_intervals.json',intervals)
 text='''# 上下文与效率补充指标

来源：服务器已有评测逐步记录，离线CPU统计；没有重新训练或评测。逐题及逐步原始指标见 results/episodes/ 下的5个JSON文件，重算入口为 tools/analyze_context_metrics.py。

输入长度采用服务返回的prompt_tokens（包含实际发送的系统提示与历史）；每次调用等权。每题峰值是该题所有调用中的最大输入长度，不是输入加输出峰值，也不是实际显存占用。不要直接与排除系统提示或采用不同分词器的论文数值比较。

| 方案 | 调用次数 | 每次输入均值 | 每次输入P95 | 每题峰值输入均值 | 每题峰值输入P95 |
|---|---:|---:|---:|---:|---:|
'''
 for key in NAMES:
  if key not in summary:continue
  s=summary[key];u=s['input_per_call'];v=s['peak_input_per_episode'];text+=f"| {s['name']} | {s['calls']} | {u['mean']:.1f} | {u['p95']:.1f} | {v['mean']:.1f} | {v['p95']:.1f} |\n"
 text+='\n## 配对差异的95%置信区间\n\n按500个task_id成对、有放回重采样20,000次，固定seed=42。差异方向为后者减前者；成功率用百分点，奖励用0～1原尺度。区间跨0表示本次题目样本下差异方向不确定，不证明等效或不劣。这不包含重新训练随机种子的波动，也不纠正历史反复查看测试集的影响。\n\n| 对比 | 成功率差（百分点）[95% CI] | 平均奖励差 [95% CI] |\n|---|---:|---:|\n'
 for b,m in PAIRS:
  vals=[v for v in intervals if v['baseline']==b and v['method']==m]
  if not vals:continue
  strs=[f"{v['difference']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]" for v in vals]
  text+=f'| {NAMES[b]} → {NAMES[m]} | {strs[0]} | {strs[1]} |\n'
 text+='\n## 历史端到端评测耗时\n\n每题elapsed_seconds包含环境reset、动作执行、模型请求及并发等待，不是纯模型推理时间。总耗时采用timing.json，只有new_episodes=500时代表完整500题。平均每题耗时不等于总耗时除以500，因为并发执行。硬件型号和其他进程负载尚未逐次核验，以下不能作为严格受控加速实验。\n\n| 方案 | 每题平均秒 | 每题P95秒 | 整批评测分钟 | 本次计时题数 | 并发 |\n|---|---:|---:|---:|---:|---:|\n'
 for key in NAMES:
  if key not in summary:continue
  s=summary[key];t=s['end_to_end_seconds'];total=s['evaluation_seconds'];text+=f"| {s['name']} | {t['mean']:.2f} | {t['p95']:.2f} | {total/60:.2f} | {s['timed_new_episodes']} | {s['workers']} |\n" if t and total else f"| {s['name']} | 缺失 | 缺失 | 缺失 | — | — |\n"
 text+='\n主20轮对照已核对：8个评测worker、vLLM max_num_seqs=8、max_num_batched_tokens=4096、prefix caching开启、BF16、seed42、服务上下文容量262144、GPU内存利用率配置0.6、eager模式；服务容量不等于训练32K。配置摘录保存在对应逐题文件的 service_config 字段。\n'
 (a.output/'REPORT.md').write_text(without_p95_columns(text),encoding='utf-8')
 print(json.dumps({k:summary[k] for k in ['grpo20','mask4_grpo20']},ensure_ascii=False))
 print(json.dumps([v for v in intervals if v['baseline']=='grpo20' and v['method']=='mask4_grpo20']))
if __name__=='__main__':main()
