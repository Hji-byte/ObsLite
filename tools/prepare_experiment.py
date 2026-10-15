"""Check the training backend automatically, then prepare the main/baseline run."""
import argparse, subprocess, sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('output','model','snapshot','eto-root','task-list','environment-python'):
  p.add_argument('--'+name,required=True)
 p.add_argument('--fast-fp32-proof', help='Optional existing report; otherwise generated automatically')
 p.add_argument('--method',choices=['mask4','full'],default='mask4')
 p.add_argument('--steps',type=int,default=20)
 a=p.parse_args()
 output=Path(a.output).resolve()
 if output.exists():
  raise ValueError('Use a new training output directory')
 proof=a.fast_fp32_proof
 if not proof:
  import tempfile
  check_parent=output.parent/'backend_checks'
  check_parent.mkdir(parents=True,exist_ok=True)
  check=Path(tempfile.mkdtemp(prefix=output.name+'-',dir=check_parent))/'check'
  subprocess.run([sys.executable,'-m','scripts.check_training_backend','--model',a.model,
                 '--data',str(Path(a.snapshot)/'teacher_sft.jsonl'),'--output',str(check),
                 '--method',a.method],check=True)
  proof=str(check/'proof.json')
 cmd=[sys.executable,'-m','scripts.prepare_rl']
 for name in ('output','model','snapshot','eto-root','task-list','environment-python'):
  cmd += ['--'+name,getattr(a,name.replace('-','_'))]
 cmd += ['--fast-fp32-proof',proof]
 cmd += ['--algorithm','grpo',
         '--context-policy','mask_old_observations' if a.method=='mask4' else 'full_history',
         '--keep-recent-history','4','--training-steps',str(a.steps),'--save-every','2',
         '--max-checkpoints',str(max(1,(a.steps+1)//2)), '--vllm-max-num-seqs','32',
         '--skip-initial-validation','--test-every','-1']
 subprocess.run(cmd,check=True)
if __name__=='__main__':main()
