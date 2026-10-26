from pathlib import Path
import json,re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
root=Path(__file__).parent; docs=[x.strip() for x in (root/'data/policies.txt').read_text().splitlines() if x.strip()]; vec=TfidfVectorizer(ngram_range=(1,2)); mat=vec.fit_transform(docs)
def guard_in(q): return not any(p in q.lower() for p in ['ignore all rules','reveal system prompt','bypass guardrail'])
def retrieve(q):
 s=cosine_similarity(vec.transform([q]),mat)[0]; return docs[int(s.argmax())]
def answer(q):
 if not guard_in(q): return 'Blocked: suspected prompt injection.'
 if re.search(r'(phone|address|personal data)',q,re.I) and re.search(r'(show|give|reveal|list)',q,re.I): return 'Blocked: personal data request.'
 ctx=retrieve(q); out=('Escalate to regional logistics lead. ' if 'delay' in q.lower() and re.search(r'\b(5[0-9]|[6-9][0-9])\b',q) else '')+ctx
 if re.search(r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b',out): return 'Blocked: possible phone number in output.'
 return out
rows=json.loads((root/'data/scenarios.json').read_text()); hits=0; blocked=0
for r in rows:
 out=answer(r['question']); good=('Escalate' in out if r['expected']=='escalated' else '2 and 8' in out if r['expected']=='temperature' else out.startswith('Blocked') if r['expected']=='blocked' else True); hits+=bool(good); blocked+=out.startswith('Blocked')
print(json.dumps({'scenarios':len(rows),'task_accuracy':round(hits/len(rows),3),'blocked_count':blocked,'guardrail':'input injection + direct PII request + output phone pattern'},indent=2))
print('Example:',answer(rows[0]['question']))
