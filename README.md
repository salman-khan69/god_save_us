from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score, classification_report, accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
acc=r2_score(y_test,y_pred)
print("R2: ",r2_score(y_test,y_pred))
dt=DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred3=dt.predict(X_test)
print("Report: ",classification_report(y_test,y_pred2))
acc3=accuracy_score(y_test,y_pred2)
print(acc3)

sc=SpectralClustering()
df['cluster_spec']=sc.fit_predict(X)
print(silhouette_score(X,df['cluster_spec']))
