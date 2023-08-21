import cudf
from cuml import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from cuml.metrics.regression import mean_squared_error
from sklearn.linear_model import LinearRegression as LR_sk
from cuml.ensemble import RandomForestRegressor as curfr
from sklearn.feature_selection import SequentialFeatureSelector as SFS

import numpy as np
import cupy as cp
import time
import pandas as pd

data = cudf.read_csv("Datasets/proc_train_desc_logP_smiles.csv",dtype="float32")
data_pandas = pd.read_csv("Datasets/proc_train_desc_logP_smiles.csv")

x_train = data.drop(columns=["smiles","logP_exp"])
y_train = data["logP_exp"]

x_train_pandas = data_pandas.drop(columns=["smiles","logP_exp"])
y_train_pandas = data_pandas["logP_exp"]

cv_groups_train = []
cv_groups_test = []

kf = KFold(n_splits=5,shuffle=False,random_state=None)

for i, (train_index,test_index) in enumerate(kf.split(x_train,y_train)):
    cv_groups_train.append((x_train.iloc[train_index],y_train[train_index]))
    cv_groups_test.append((x_train.iloc[test_index],y_train[test_index]))

def cross_validation_gpu(model,cv_groups_train,cv_groups_test):
    score = 0
    for i in range(len(cv_groups_train)):
        model.fit(cv_groups_train[i][0],cv_groups_train[i][1])
        y_pred = model.predict(cv_groups_test[i][0])
        score += mean_squared_error(cv_groups_test[i][1],y_pred,squared=False)
    return score/len(cv_groups_train)


#gpuModel = LinearRegression(algorithm="svd-qr")

gpuModel = curfr(n_estimators=100,max_depth=5,accuracy_metric="mse",random_state=3006,n_streams=1)

"""
fs = SFS(gpuModel,n_features_to_select="auto",tol=0.0001,direction="forward",scoring="neg_root_mean_squared_error",cv=5)

startTime = time.time()
#print(cross_validation_gpu(gpuModel,cv_groups_train,cv_groups_test))

fs.fit(x_train.to_pandas(),y_train.to_pandas())
finalTime = time.time()
print(finalTime-startTime)
features = fs.get_feature_names_out()
print(np.mean(cross_val_score(gpuModel,x_train.to_pandas()[features],y_train.to_pandas(),cv=5,scoring="neg_root_mean_squared_error")))
"""