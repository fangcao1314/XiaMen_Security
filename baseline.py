# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:26:15 2019

@author: jack
"""

from xgboost import XGBClassifier
import pandas  as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt



# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
train_x = r"train.csv"
train_y = r"train_target.csv"
test_path = r"test.csv"

train_x = pd.read_csv(train_x,index_col="id")
train_y = pd.read_csv(train_y,index_col="id")
train = pd.merge(train_x,train_y,on='id')



colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#train.to_csv('train_xy.csv')
train_target_0 = train[train.target==0]
train_target_1 = train[train.target==1]
print(train_target_0.shape)
print(train_target_1.shape)

test_data = pd.read_csv(test_path,index_col="id")


xgb = XGBClassifier()

res = pd.DataFrame(index=test_data.index)
#train_p = pd.DataFrame(index=train_data.index)

def train(ite):
    print(i)
    data = train_target_0.sample(700)#数据显示1 ：0 = 17：2（》0.5）
    data = data.append(train_target_1)
    y_ = data.target
    del data['target']
    xgb.fit(data,y_)
#    train_p[ite] = xgb.predict(train_data)
    res[ite] = xgb.predict_proba(test_data)[:,1]
    

for i in range(300):
    train(i)
    
resT = res.T
a = resT.apply(sum)
a/300
b= a/300
c=pd.DataFrame(b)
c.to_csv('result3.csv',index=False)