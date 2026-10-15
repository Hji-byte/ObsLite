import argparse
from pathlib import Path
from scripts.export_model import export_model
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--model',type=Path,required=True)
 p.add_argument('--adapter',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();export_model(a.model,a.adapter,a.output)
 from transformers import GenerationConfig
 GenerationConfig.from_pretrained(a.model,local_files_only=True).save_pretrained(a.output)
 from transformers import GenerationConfig
 GenerationConfig.from_pretrained(a.model,local_files_only=True).save_pretrained(a.output)
