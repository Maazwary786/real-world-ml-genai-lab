import subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class TaskSmokeTests(unittest.TestCase):
 def test_all_task_entrypoints(self):
  entries=['main.py']*8+['prepare.py','main.py']
  for i,entry in enumerate(entries,1):
   with self.subTest(task=i):
    result=subprocess.run([sys.executable,str(ROOT/f'task{i:02d}'/entry)],cwd=ROOT/f'task{i:02d}',text=True,capture_output=True,timeout=90)
    self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
    self.assertTrue(result.stdout.strip())
 def test_exactly_ten_task_folders(self):
  self.assertEqual([f'task{i:02d}' for i in range(1,11)],[p.name for p in sorted(ROOT.glob('task*'))])
if __name__=='__main__': unittest.main()
