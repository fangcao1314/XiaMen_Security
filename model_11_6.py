# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:58:03 2019

@author: jack
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:26:15 2019

@author: jack
"""
from tqdm import tqdm
from xgboost import XGBClassifier
import pandas  as pd
from imblearn.over_sampling import SMOTE
file_dir = ''
train_x = file_dir + r"train.csv"
train_y = file_dir + r"train_target.csv"
test_path = file_dir + r"test.csv"

train_x = pd.read_csv(train_x,index_col="id")
train_y = pd.read_csv(train_y,index_col="id")


train = pd.merge(train_x,train_y,on='id')
print('训练数据：',train.shape)

def cut_age(x):
    if x<20:
        return 0
    elif x>=20 and x<33:
        return 1
    elif x>=33 and x<45:
        return 2
    else:
        return 3
train = pd.concat([train, pd.DataFrame(columns=['sheng','shi','sheng_local','shi_local','islocal_shi','islocal_sheng','time_all'])])
train['age'] = train['age'].apply(lambda x : cut_age(x) )
train['sheng']= train['dist'].apply(lambda x:int(str(x)[:2]))
train['shi'] = train['dist'].apply(lambda x:int(str(x)[2:4]))
train['sheng_local'] = train['residentAddr'].apply(lambda x: int(str(x)[:2]))
train['shi_local'] = train['residentAddr'].apply(lambda x: int(str(x)[2:4]))
#train = pd.merge()
#train = pd.concat([train,sheng,shi,sheng_local,shi_local])
train['edu'] = train['edu'].apply(lambda x:x//10)
def islocal_(x,y):
    if(x==y):
        return 1
    else:
        return 0
    
train['islocal_sheng']= train.apply(lambda row:islocal_(row['sheng'],row['sheng_local']),axis=1)
train['islocal_shi'] = train.apply(lambda row:islocal_(row['shi'],row['shi_local']),axis=1)

train['age_bin'] = pd.cut(train['age'],20,labels = False)
train = train.drop(['age'],axis = 1)
train["lmt_bin"] = pd.qcut(train["lmt"],50,labels=False)
train = train.drop(['lmt'], axis=1)
train["dist_bin"] = pd.qcut(train["dist"],60,labels=False)
train = train.drop(['dist'], axis=1)
train['time_all'] = train['certValidStop'] - train['certValidBegin']
train["time_all_bin"] = pd.cut(train["time_all"],30,labels=False)
train = train.drop(['time_all'],axis = 1)
del train['bankCard']




test_data = pd.read_csv(test_path,index_col="id")
print('测试数据：',test_data.shape)
del test_data['bankCard']
test_data = pd.concat([test_data, pd.DataFrame(columns=['sheng','shi','sheng_local','shi_local','islocal_shi','islocal_sheng'])])
test_data['age'] = test_data['age'].apply(lambda x : cut_age(x) )
test_data['sheng']= test_data['dist'].apply(lambda x:int(str(x)[:2]))
test_data['shi'] = test_data['dist'].apply(lambda x:int(str(x)[2:4]))
test_data['sheng_local'] = test_data['residentAddr'].apply(lambda x: int(str(x)[:2]))
test_data['shi_local'] = test_data['residentAddr'].apply(lambda x: int(str(x)[2:4]))
#train = pd.merge()
#train = pd.concat([train,sheng,shi,sheng_local,shi_local])
test_data['edu'] = test_data['edu'].apply(lambda x:x//10)
test_data['islocal_sheng']= test_data.apply(lambda row:islocal_(row['sheng'],row['sheng_local']),axis=1)
test_data['islocal_shi'] = test_data.apply(lambda row:islocal_(row['shi'],row['shi_local']),axis=1)


test_data['age_bin'] = pd.cut(test_data['age'],20,labels = False)
test_data = test_data.drop(['age'],axis = 1)
test_data["lmt_bin"] = pd.qcut(test_data["lmt"],50,labels=False)
test_data = test_data.drop(['lmt'], axis=1)
test_data["dist_bin"] = pd.qcut(test_data["dist"],60,labels=False,duplicates = 'drop')
test_data = test_data.drop(['dist'], axis=1)
test_data['time_all'] = test_data['certValidStop'] - test_data['certValidBegin']
test_data["time_all_bin"] = pd.cut(test_data["time_all"],30,labels=False)
test_data = test_data.drop(['time_all'],axis = 1)


#
#train_data = train
#test_data = test_data
#
#dummy_fea = ["gender","job", "loanProduct", "basicLevel","ethnic"]
#train_test_data = pd.concat([train,test_data],axis=0,ignore_index = False) 
#dummy_df = pd.get_dummies(train.loc[:,dummy_fea], columns=train.loc[:,dummy_fea].columns)
#dunmy_fea_rename_dict = {}
#for per_i in dummy_df.columns.values:
#    dunmy_fea_rename_dict[per_i] = per_i + '_onehot'
#print (">>>>>",  dunmy_fea_rename_dict)
#dummy_df = dummy_df.rename( columns=dunmy_fea_rename_dict )
#train_test_data = pd.concat([train_test_data,dummy_df],axis=1)
#column_headers = list( train_test_data.columns.values )
#print(column_headers)
#train_test_data = train_test_data.drop(dummy_fea,axis=1)
#column_headers = list( train_test_data.columns.values )
#print(column_headers)
#train_train = train_test_data.iloc[:train_data.shape[0],:]
#test_test = train_test_data.iloc[train_data.shape[0]:,:]
#train = train_train
#test_data = test_test


train_target_0 = train[train.target==0]
train_target_1 = train[train.target==1]

print (train.shape)
print (test_data.shape)
#del test_data['target']

res = pd.DataFrame(index=test_data.index)

xgb = XGBClassifier()
#train_p = pd.DataFrame(index=train_data.index)

def train(ite):
    data = train_target_0.sample(900)#数据显示1 ：0 = 17：2（》0.5）
    data = data.append(train_target_1)
    y_ = data.target
    del data['target']
    xgb.fit(data, y_)
    #train_p[ite] = xgb.predict(train_data)
    res[ite] = xgb.predict_proba(test_data)[:,1]

iter_num = 300

for i in tqdm(range(iter_num)):
    train(i)
    
resT = res.T
print(resT.shape)
result = resT.apply(sum) / iter_num
save_file = pd.DataFrame(result)
save_file.rename(columns={0:"target"},inplace=True)
save_file.index.name = 'id'
print(save_file)
save_file.to_csv('result.csv')
print("complete")