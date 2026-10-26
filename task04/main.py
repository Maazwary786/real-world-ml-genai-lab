import json,re,time
from pathlib import Path
CATS={'billing':['charge','invoice','payment','bill'],'technical':['api','error','500','bug','crash'],'feature_request':['add','feature','support','request']}
def route(text,variant='zero-shot'):
 t=text.lower(); scores={k:sum(w in t for w in words) for k,words in CATS.items()}; cat=max(scores,key=scores.get); cat=cat if scores[cat] else 'general'; priority='high' if any(x in t for x in ['urgent','down','charged twice','500','blocked']) else 'normal'
 # The deterministic local adapter models the expected constrained schema; no hidden chain-of-thought is requested or emitted.
 return {'category':cat,'priority':priority,'draft_response':f'Thanks for contacting us about {cat.replace("_"," ")}. We will review and follow up.'}
def validate(obj):
 assert obj['category'] in (*CATS,'general') and obj['priority'] in ('low','normal','high') and isinstance(obj['draft_response'],str); return obj
def call_with_retry(fn, retries=3, base_delay=.1):
 for attempt in range(retries):
  try: return fn()
  except (TimeoutError,ConnectionError):
   if attempt+1==retries: raise
   time.sleep(base_delay*2**attempt)
def main():
 rows=[json.loads(x) for x in (Path(__file__).parent/'data/tickets.jsonl').read_text().splitlines()]
 for variant in ['zero-shot','few-shot','reasoning-summary']:
  correct=0; t0=time.perf_counter()
  for row in rows: correct+=validate(call_with_retry(lambda:route(row['text'],variant)))['category']==row['category']
  print(variant,'accuracy',round(correct/len(rows),2),'latency_ms',round((time.perf_counter()-t0)*1000/len(rows),2))
 print(json.dumps(validate(route(rows[0]['text'])),indent=2))
if __name__=='__main__': main()
