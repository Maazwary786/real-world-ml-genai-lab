from pathlib import Path
import ast,re,json
source=(Path(__file__).parent/'data/vulnerable.py').read_text(); log=[]
findings=[]
if re.search(r'\beval\s*\(',source): findings.append({'severity':'high','rule':'unsafe-eval','line':next(i for i,l in enumerate(source.splitlines(),1) if 'eval(' in l),'recommendation':'Replace dynamic evaluation with a safe parser.'})
log.append({'agent':'security_scanner','findings':findings}); patch=source.replace('return eval(user_input)','return user_input.strip()'); log.append({'agent':'remediation','patch':'Replace eval with inert string handling (demo remediation).'});
try: ast.parse(patch); qa={'passed':True,'syntax':'valid'}
except SyntaxError as e: qa={'passed':False,'error':str(e)}
log.append({'agent':'qa_reviewer','result':qa}); print(json.dumps(log,indent=2)); print('Demo patch preview:\n'+patch)
