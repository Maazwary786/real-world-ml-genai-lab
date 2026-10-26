import ast,operator,sqlite3
from pathlib import Path
ROOT=Path(__file__).parent
def sql_portfolio(customer):
 """Read portfolio balances for one customer using a parameterized query."""
 with sqlite3.connect(ROOT/'data/portfolio.db') as db: return db.execute('select asset,amount from balances where customer=?',(customer,)).fetchall()
def market_search(query):
 """Retrieve the local market outlook passage most relevant to a query."""
 return (ROOT/'data/market.txt').read_text().strip()
def calculate(expression):
 """Evaluate a basic arithmetic expression with a strict AST allowlist."""
 ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}; node=ast.parse(expression,mode='eval').body
 def go(n):
  if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
  if isinstance(n,ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](go(n.left),go(n.right))
  raise ValueError('Only basic arithmetic is allowed')
 return go(node)
def agent(customer='Customer X'):
 trace=[]; rows=sql_portfolio(customer); trace.append(('Action','sql_portfolio',rows)); outlook=market_search('Q3 tax market risk'); trace.append(('Action','market_search',outlook)); total=calculate('+'.join(str(a) for _,a in rows)); trace.append(('Action','calculator',total)); answer=f'{customer} portfolio total: ${total:,.0f}. Market context: {outlook} Projected tax risk requires basis, holding period, jurisdiction, and professional review.'
 for event,tool,result in trace: print(f'{event} -> {tool} -> {result}')
 print('Final:',answer)
if __name__=='__main__': agent()
