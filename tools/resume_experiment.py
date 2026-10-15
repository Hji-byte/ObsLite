"""Prepare explicit checkpoint continuation; preserve training semantics."""
import argparse,json,shutil
from pathlib import Path
import yaml
from scripts.common import sha256,write_json
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True)
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--steps',type=int,default=20)
 a=p.parse_args();src=a.source.resolve();out=a.output.resolve();ck=a.checkpoint.resolve()
 if out.exists():raise ValueError('Output must not already exist')
 for name in ['data.pt','actor/model_world_size_1_rank_0.pt','actor/optim_world_size_1_rank_0.pt','actor/extra_state_world_size_1_rank_0.pt']:
  if not (ck/name).is_file():raise ValueError('Missing checkpoint file: '+name)
 m=json.loads((src/'manifest.json').read_text())
 for name,digest in m['prepared_files'].items():
  if sha256(src/name)!=digest:raise ValueError('Source manifest mismatch: '+name)
 c=yaml.safe_load((src/'config/ppo_trainer.yaml').read_text())
 old=int(ck.name.split('_')[-1])
 if a.steps<=old:raise ValueError('Final step must exceed checkpoint step')
 if c['webshop_adapter'].get('pods') or c['webshop_adapter'].get('spear'):raise ValueError('This entry is for plain GRPO only')
 out.mkdir();(out/'config').mkdir()
 # Dataset paths intentionally keep pointing at immutable original prepared data.
 for name in m['prepared_files']:
  q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/name,q)
 c['trainer'].update(total_training_steps=a.steps,save_freq=2,max_actor_ckpt_to_keep=max(1,(a.steps-old+1)//2),resume_mode='resume_path',resume_from_path=str(ck),default_local_dir=str(out/'checkpoints'))
 (out/'config/ppo_trainer.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
 m['prepared_files']={name:sha256(out/name) for name in m['prepared_files']}
 m['continuation']={'source':str(src),'checkpoint':str(ck),'last_step':a.steps,'save_every':2,'training_parameters_changed':False}
 write_json(out/'manifest.json',m)
if __name__=='__main__':main()
