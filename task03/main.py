from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(Path(__file__).parent/'data/demand.csv'); X=df.drop(columns='demand'); y=df.demand; a,b,c,d=train_test_split(X,y,test_size=.2,random_state=1)
search=RandomizedSearchCV(make_pipeline(StandardScaler(),MLPRegressor(max_iter=350,early_stopping=True,random_state=1)),{'mlpregressor__hidden_layer_sizes':[(32,),(64,32),(64,32,16)],'mlpregressor__alpha':[.0001,.01,.1],'mlpregressor__learning_rate_init':[.001,.01]},n_iter=5,cv=3,random_state=1,n_jobs=1).fit(a,c)
for name,m in [('MLP tuned',search.best_estimator_),('RF baseline',RandomForestRegressor(n_estimators=100,random_state=1).fit(a,c))]:
 pred=m.predict(b); print(name,'MAE',round(mean_absolute_error(d,pred),2),'RMSE',round(mean_squared_error(d,pred)**.5,2))
print('best',search.best_params_)
try:
 import torch; print('Optional PyTorch installed. sklearn MLP above is the runnable CPU-first training path.')
except ImportError: print('PyTorch absent; using sklearn MLP to keep the demo lightweight.')
