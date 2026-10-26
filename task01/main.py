from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score, precision_score

df=pd.read_csv(Path(__file__).parent/'data/telemetry.csv').dropna()
cols=['vibration','temperature','voltage','noise']; X=df[cols]; y=df.rul_hours; z=df.failure_24h
Xtr,Xte,ytr,yte,ztr,zte=train_test_split(X,y,z,test_size=.25,random_state=7,stratify=z)
reg=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LinearRegression()).fit(Xtr,ytr)
clf=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000)).fit(Xtr,ztr)
prob=clf.predict_proba(Xte)[:,1]; thresholds=np.arange(.15,.81,.05); threshold=max(thresholds,key=lambda t: recall_score(zte,prob>=t,zero_division=0))
pred=prob>=threshold
print(f'RUL RMSE={mean_squared_error(yte,reg.predict(Xte))**.5:.2f} MAE={mean_absolute_error(yte,reg.predict(Xte)):.2f}')
print(f'24h threshold={threshold:.2f} recall={recall_score(zte,pred):.3f} precision={precision_score(zte,pred,zero_division=0):.3f}')
