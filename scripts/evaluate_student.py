"""Official WebShop test IDs 0..499, same runtime and prompt for base and SFT."""
import argparse
import json
import os
from pathlib import Path
import uuid
import importlib.metadata
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
from .common import DEFAULT_ETO, read_json, read_jsonl, write_json, sha256
from .generate_teacher import (create_environment, verify_upstream, load_prompt,
                               run_lock, ensure_manifest, run_episode, summarize)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher-snapshot',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/datasets/webshop/teacher_train_2000_seed42_no_demo/tasks.jsonl'))
    p.add_argument('--eto-root',type=Path,default=DEFAULT_ETO)
    p.add_argument('--base-url',default='http://127.0.0.1:8018')
    p.add_argument('--count',type=int,default=500)
    p.add_argument('--check-env',action='store_true')
    p.add_argument('--no-demo',action='store_true',help='Use SFT-aligned system-only prefix; no fixed demonstration')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--context-policy',choices=('full_history','mask_old_observations','recent_action_window'),
                   default='full_history')
    p.add_argument('--keep-recent-history',type=int,default=4,
                   help='Prior observations kept in addition to the task and current observation')
    p.add_argument('--server-info',type=Path,help='Audited vLLM service metadata')
    p.add_argument('--reuse-evaluation',type=Path,help='User-authorized backend migration; preserve completed episodes')
    args=p.parse_args()
    if args.context_policy == 'recent_action_window' and not args.no_demo:
        p.error('recent_action_window requires --no-demo')
    assert 1<=args.count<=500
    assert 1<=args.workers<=8
    from .context_policy import context_policy_from_config
    context_policy=context_policy_from_config({'type':args.context_policy,
        'keep_recent_history':args.keep_recent_history,'preserve_error_observations':True})
    upstream=verify_upstream(args.eto_root)
    prefix=load_prompt(args.eto_root)
    # load_prompt reads the shared scripts/prompts/react_webshop.json.
    if args.no_demo:
        from .no_demo_prompt import no_demo_prefix
        prefix=no_demo_prefix(prefix,args.teacher_snapshot)
    rows=read_jsonl(args.dataset)
    with run_lock(args.output):
        env,assets=create_environment(args.eto_root,rows,42)
        try:
            tests=[{'task_id':i,'instruction_reference':env.server.goals[i]['instruction_text']} for i in range(args.count)]
            write_json(args.output/'test_tasks.json',tests)
            if args.check_env:
                write_json(args.output/'environment_check.json',{'passed':True,'test_count':len(tests),'assets':assets})
                print('Full environment and official test split verified',flush=True)
                return
            if args.server_info:
                info=read_json(args.server_info)
                models=json.load(urlopen(args.base_url+'/v1/models',timeout=10))['data']
                assert any(m['id']=='webshop-08b' and m.get('root')==info['export_path'] for m in models)
            else:
                info=json.load(urlopen(args.base_url+'/info',timeout=10))
            runtime={'python':platform.python_version(), 'packages':{
                name:importlib.metadata.version(name) for name in
                ('torch','spacy','en_core_web_lg','pyserini','gym','numpy','beautifulsoup4','thefuzz')}}
            config={'task_ids':list(range(args.count)),'temperature':0,'top_p':1,'max_steps':20,'max_tokens':1024,
                    'seed':42,'prompt':prefix,'upstream':upstream,'assets':assets,'server':info,
                    'runtime':runtime,'workers':args.workers,
                    'evaluator_sha256':sha256(Path(__file__)),'runner_sha256':sha256(Path(__file__).with_name('generate_teacher.py'))}
            if context_policy is not None:
                config['context_policy']=context_policy.to_config()
                config['context_policy_sha256']=sha256(Path(__file__).with_name('context_policy.py'))
            if args.reuse_evaluation:
                assert args.server_info, 'Backend migration requires audited vLLM metadata'
                source=args.reuse_evaluation
                config['reuse_source_manifest_sha256']=sha256(source/'manifest.json')
                config['reuse_source']=str(source.resolve())
                old=read_json(source/'manifest.json')
                allowed={'server','evaluator_sha256','reuse_source','reuse_source_manifest_sha256','workers'}
                assert all(old.get(k)==config.get(k) for k in set(old)|set(config) if k not in allowed), 'Evaluation protocol mismatch'
                assert old['server']['model_hashes']==info['model_hashes']
                assert old['server']['template']==info['template']
                assert old['server']['adapter'] is None and info['adapter'] is None
                if old['workers']!=args.workers:
                    assert old['server'].get('backend')=='vllm' and info['backend']=='vllm'
                    assert all(old['server'].get(k)==info.get(k) for k in set(old['server'])|set(info) if k!='max_num_seqs')
                    assert old['workers']==old['server']['max_num_seqs'] and args.workers==info['max_num_seqs']
            ensure_manifest(args.output,config)
            if args.reuse_evaluation:
                reused=[]
                with run_lock(source):
                    for row in tests:
                        path=source/'episodes'/f"{row['task_id']}.json"
                        if not path.exists(): continue
                        episode=read_json(path)
                        assert episode['task_id']==row['task_id'] and episode['status']=='completed'
                        target=args.output/'episodes'/path.name
                        if target.exists(): assert read_json(target)==episode
                        else: write_json(target,episode)
                        reused.append({'task_id':row['task_id'],'source_episode_sha256':sha256(path)})
                write_json(args.output/'backend_reuse.json',{'source':str(source),'source_server':old['server'],
                    'new_server':info,'reused':reused,
                    'parent_reuse':read_json(source/'backend_reuse.json') if (source/'backend_reuse.json').exists() else None,
                    'policy':'User authorized backend/concurrency continuation; completed successes and failures preserved.'})
                print(json.dumps(summarize(args.output,tests,export=False)),flush=True)
            from openai import OpenAI
            environment_lock=threading.RLock()
            local=threading.local()
            environments=[]
            class LockedEnvironment:
                def __init__(self,wrapped): self.wrapped=wrapped
                def __getattr__(self,name):
                    with environment_lock:
                        value=getattr(self.wrapped,name)
                    if not callable(value): return value
                    def call(*a,**kw):
                        with environment_lock: return value(*a,**kw)
                    return call

            def evaluate_one(row):
                # Only environment operations are serialized. Model requests overlap.
                if not hasattr(local,'env'):
                    with environment_lock:
                        worker_env=type(env)(observation_mode='text_rich',server=env.server,
                                             session_prefix='parallel_'+uuid.uuid4().hex+'_')
                        environments.append(worker_env)
                        local.env=LockedEnvironment(worker_env)
                target=args.output/'episodes'/f"{row['task_id']}.json"
                attempt=args.output/'attempts'/f"{row['task_id']}_{uuid.uuid4().hex}.jsonl"
                attempt.parent.mkdir(parents=True,exist_ok=True)
                with OpenAI(api_key='local',base_url=args.base_url+'/v1',timeout=600,max_retries=0) as client, attempt.open('a',encoding='utf-8') as stream:
                    def event(value):
                        stream.write(json.dumps(value,ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    try:
                        result=run_episode(local.env,lambda messages: complete(client,messages),row,prefix,20,event,
                                           context_policy=context_policy)
                    except Exception as exc:
                        event({'kind':'error','type':type(exc).__name__})
                        raise RuntimeError(f'Evaluation interrupted at task {row["task_id"]}: {type(exc).__name__}') from exc
                return target,result

            def complete(client,messages):
                    extra={'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}} if args.server_info else {}
                    result=client.chat.completions.create(model='webshop-08b',messages=messages,
                        temperature=0,top_p=1,max_tokens=1024,**extra)
                    choice=result.choices[0]
                    return {'content':choice.message.content or '', 'finish_reason':choice.finish_reason,
                            'usage':result.usage.model_dump(),'response_id':result.id,'response_model':result.model}
            started=time.monotonic()
            pending_rows=[row for row in tests if not (args.output/'episodes'/f"{row['task_id']}.json").exists()]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(evaluate_one,row) for row in pending_rows]
                for future in as_completed(futures):
                    try:
                        target,result=future.result()
                    except BaseException:
                        for other in futures: other.cancel()
                        raise
                    result.pop('sft_messages',None)  # Test rollouts are never training data.
                    write_json(target,result)
                    summary=summarize(args.output,tests,export=False)
                    print(json.dumps(summary),flush=True)
            write_json(args.output/'timing.json',{'evaluation_seconds':time.monotonic()-started,
                'new_episodes':len(pending_rows),'workers':args.workers})
        finally:
            env.close()


if __name__=='__main__': main()

