from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score,silhouette_score
df=pd.read_csv(Path(__file__).parent/'data/transactions.csv'); X=df[['amount','location_delta','velocity','account_age_days']]; y=df.fraud
a,b,c,d=train_test_split(X,y,test_size=.3,random_state=3,stratify=y)
for name,model in [('DecisionTree',DecisionTreeClassifier(max_depth=4,class_weight='balanced',random_state=3)),('RandomForest',RandomForestClassifier(n_estimators=100,max_depth=7,class_weight='balanced',random_state=3))]:
 model.fit(a,c); print(name,'ROC-AUC',round(roc_auc_score(d,model.predict_proba(b)[:,1]),3))
 if name=='DecisionTree': print(export_text(model,feature_names=list(X.columns)))
profiles=StandardScaler().fit_transform(df[['user_spend','user_velocity']]); print('k, inertia, silhouette')
for k in range(2,6):
 m=KMeans(n_clusters=k,random_state=3,n_init=10).fit(profiles); print(k,round(m.inertia_,2),round(silhouette_score(profiles,m.labels_),3))
