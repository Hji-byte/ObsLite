"""Explicit-path version of the historical RL evaluation server and 500-task runner."""
import argparse,json,os,signal,socket,subprocess,time
from pathlib import Path
from contextlib import contextmanager
from urllib.request import Request,urlopen
from scripts.common import ROOT,read_json,write_json,sha256
def gpu_idle(timeout=90):
    until = time.monotonic()+timeout
    while True:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if not active:
            return
        if time.monotonic() >= until:
            raise RuntimeError(f'GPU processes remain active; refusing overlap: {active}')
        time.sleep(2)
def service_metadata(model, info_template=None):
    return read_json(info_template) if info_template else {
        'template': sha256(model/'chat_template.jinja'),
        'max_num_seqs': 8, 'max_num_batched_tokens': 4096,
        'max_model_len': 262144, 'prefix_caching': True,
        'backend': 'vllm', 'dtype': 'bfloat16', 'seed': 42,
        'gpu_memory_utilization': 0.6, 'enforce_eager': True,
        'mamba_cache_mode': 'align', 'enable_thinking': False,
        'model_hashes': {p.name: sha256(p) for p in sorted(model.glob('*.safetensors'))},
        'model': str(model), 'metadata_source': 'generated from model and evaluator settings'}

@contextmanager
def evaluation_server(model, adapter, python, info_template, directory, env):
    """Retain the original SFT500 evaluator/server settings; add the RL LoRA."""
    directory.mkdir(parents=True, exist_ok=False)
    adapter = adapter.resolve() if adapter else None
    info = service_metadata(model, info_template)
    assert info['max_num_seqs'] == 8 and info['max_num_batched_tokens'] == 4096
    assert info['max_model_len'] == 262144 and info['prefix_caching'] is True
    assert info['template'] == sha256(model/'chat_template.jinja')
    if adapter:
        info.update(adapter=str(adapter), export_path=str(adapter), base_export=str(model),
            inference_mode='frozen merged SFT plus checkpoint RL LoRA; no merge rounding',
            adapter_hashes={p.name: sha256(p) for p in adapter.iterdir() if p.is_file()},
            base_export_weights_sha256=sha256(model/'model.safetensors'))
    else:
        info.update(adapter=None, export_path=str(model), base_export=str(model),
            inference_mode='frozen merged SFT without RL adapter',
            base_export_weights_sha256=sha256(model/'model.safetensors'))
        info.pop('adapter_hashes', None)
    write_json(directory/'server_info.json', info)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8020))
    args = [str(python), '-m', 'vllm.entrypoints.cli.main', 'serve', str(model),
        '--served-model-name', 'webshop-sft' if adapter else 'webshop-08b', '--host', '127.0.0.1', '--port', '8020',
        '--dtype', 'bfloat16', '--seed', '42', '--max-model-len', '262144',
        '--max-num-seqs', '8', '--max-num-batched-tokens', '4096',
        '--gpu-memory-utilization', '0.60', '--enable-prefix-caching',
        '--mamba-cache-mode', 'align', '--enforce-eager', '--generation-config', 'vllm']
    if adapter:
        args += ['--enable-lora', '--max-lora-rank', '64', '--lora-modules', f'webshop-08b={adapter}']
    server_env = dict(env, VLLM_WORKER_MULTIPROC_METHOD='spawn')
    with (directory/'server.log').open('w') as log:
        server = subprocess.Popen(args, cwd=ROOT, env=server_env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        write_json(directory/'server_process.json', {'pid': server.pid, 'command': args})
        try:
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError('Evaluation server exited; inspect server.log')
                try:
                    with urlopen('http://127.0.0.1:8020/health', timeout=2):
                        break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError('Evaluation server readiness timeout')
            models = json.load(urlopen('http://127.0.0.1:8020/v1/models', timeout=10))['data']
            assert any(m['id'] == 'webshop-08b' and m['root'] == str(adapter or model) and (not adapter or m['parent'] == 'webshop-sft') for m in models)
            # Exercises actual LoRA loading, independently of registry readiness.
            request = Request('http://127.0.0.1:8020/v1/chat/completions',
                data=json.dumps({'model': 'webshop-08b', 'messages': [{'role':'user','content':'Reply OK.'}],
                    'temperature': 0, 'max_tokens': 1,
                    'chat_template_kwargs': {'enable_thinking': False}}).encode(),
                headers={'Content-Type': 'application/json'})
            response = json.load(urlopen(request, timeout=180))
            assert response['choices'] and response['usage']['completion_tokens'] > 0
            write_json(directory/'api_preflight.json', {'passed': True, 'models': models,
                'response_model': response['model'], 'usage': response['usage'], 'heldout_tasks_used': 0})
            yield directory/'server_info.json'
        finally:
            # Only signal the private process group created immediately above.
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    gpu_idle()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['model','model-python','environment-python','snapshot','dataset','eto-root','output']:
  p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--info-template',type=Path,help='Optional existing service metadata')
 p.add_argument('--adapter',type=Path,help='RL adapter; omit for frozen SFT evaluation')
 p.add_argument('--context-policy',choices=['full_history','mask_old_observations','recent_action_window'],default='mask_old_observations')
 p.add_argument('--keep-recent-history',type=int,default=4)
 p.add_argument('--run',action='store_true')
 a=p.parse_args()
 if not a.run:
  print('Plan only: 500 tasks, greedy, 20 steps, 1024 response tokens; add --run to execute.');return
 gpu_idle(timeout=0)
 env=dict(os.environ,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',VLLM_USE_FLASHINFER_SAMPLER='0')
 with evaluation_server(a.model.resolve(),a.adapter.resolve() if a.adapter else None,a.model_python.resolve(),(a.info_template.resolve() if a.info_template else None),a.output/'server',env) as info:
  subprocess.run([str(a.environment_python),'-m','scripts.evaluate_student','--teacher-snapshot',str(a.snapshot),'--dataset',str(a.dataset),'--eto-root',str(a.eto_root),'--output',str(a.output/'evaluation'),'--no-demo','--count','500','--workers','8','--base-url','http://127.0.0.1:8020','--server-info',str(info),'--context-policy',a.context_policy,'--keep-recent-history',str(a.keep_recent_history)],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
