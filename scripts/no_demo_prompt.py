"""Select and audit the no-demo prefix against the frozen SFT data."""
import json

def no_demo_prefix(prefix, snapshot):
    assert prefix and prefix[0]['role']=='system'
    count=0
    with (snapshot/'teacher_sft.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            assert row['messages'][0]==prefix[0], 'SFT system instruction mismatch'
            assert row['messages'][1]['role']=='user'
            count+=1
    assert count>0
    return [dict(prefix[0])]
