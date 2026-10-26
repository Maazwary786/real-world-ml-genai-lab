import json
from pathlib import Path
rows=[json.loads(x) for x in (Path(__file__).parent/'data/claims.jsonl').read_text().splitlines()]
assert all(set(x)=={'system','user','assistant'} for x in rows)
print('Prepared',len(rows),'anonymized instruction examples. Placeholder IDs only; do not train on identifiable claims.')
print('Example roles:',list(rows[0]))
try:
 import torch,transformers,peft,trl,bitsandbytes
 print('Optional QLoRA stack present. See train_qlora.py and set BASE_MODEL explicitly.')
except ImportError as e: print('No training dependencies installed; lightweight prep-only mode:',e)
