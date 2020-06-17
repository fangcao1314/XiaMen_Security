# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:33:43 2019

@author: jack
"""

import pandas as pd 
import numpy as np
from sklearn import preprocessing

train_file = 'train.csv'
train_file_target = 'train_target.csv'

test_file = 'test.csv'
test_file_target = 'test_target.csv'

cols = ['loanProduct','basicLevel',
        'gender','ncloseCreditCard','unpayIndvLoan','unpayOtherLoan','unpayNormalLoan','5yearBadloan',
        'age','dist','edu','job','lmt','ethnic','residentAddr','highestEdu','linkRela',
       ]

def data_preprocessing(dataFrame):

    #fill default values
    #boolean
    dataFrame['gender'] = dataFrame['gender'].map({1:0, 2:1}).astype(int)
    dataFrame['ncloseCreditCard'] = dataFrame['ncloseCreditCard'].map({0:0, 1:1,-999:0}).astype(int)
    dataFrame['unpayIndvLoan'] = dataFrame['unpayIndvLoan'].map({0:0, 1:1, -999:0}).astype(int)
    dataFrame['unpayOtherLoan'] = dataFrame['unpayOtherLoan'].map({0:0, 1:1, -999:0}).astype(int)
    dataFrame['unpayNormalLoan'] = dataFrame['unpayNormalLoan'].map({0:0, 1:1, -999:0}).astype(int)
    dataFrame['5yearBadloan'] = dataFrame['5yearBadloan'].map({0:0, 1:1, -999:0}).astype(int)
    #float
    dataFrame['highestEdu'] = dataFrame['highestEdu'].replace({-999:0},inplace=False)
    dataFrame['residentAddr'] = dataFrame['residentAddr'].replace({-999:0},inplace=False)
    dataFrame['linkRela'] = dataFrame['linkRela'].replace({-999:0},inplace=False)
    dataFrame['basicLevel'] = dataFrame['basicLevel'].replace({-999:0},inplace=False)

    one_hot_loanProduct = pd.get_dummies(dataFrame['loanProduct'], prefix='loanProduct')
    one_hot_basiclevel = pd.get_dummies(dataFrame['basicLevel'], prefix='basicLevel')
    del dataFrame['loanProduct']
    del dataFrame['basicLevel']
    dataFrame = pd.concat([dataFrame,one_hot_loanProduct,one_hot_basiclevel],axis=1)

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = pd.DataFrame(minmax_scale.fit_transform(dataFrame))
    scaledFeatures.index=range(1,155590+1)
    scaledFeatures.index.name='id'
    #print(dataFrame.columns.values)
    #print(scaledFeatures.loc[:30,:])
    return scaledFeatures

#load data
train_features = pd.read_csv(train_file,index_col='id').loc[:,cols]
train_labels = pd.read_csv(train_file_target,index_col='id').loc[:,'target']
test_features = pd.read_csv(test_file,index_col='id').loc[:,cols]

#process data
features = pd.concat((train_features,test_features))
features = data_preprocessing(features)

#prepare datasets
train_features = features.loc[1:train_features.shape[0],:] #id: 1~132029
test_fectures = features.loc[train_features.shape[0]+1:,:] #id: 132030~155590
train_labels = pd.DataFrame(train_labels) #id: 1~132029

#simulate fake_samples
simulate_data = pd.merge(train_features,train_labels,on='id')
simulate_data_0 = simulate_data[simulate_data.target==0] #131070 *24+1(target)
simulate_data_1 = simulate_data[simulate_data.target==1] #959 *24+1(target)

train_features = simulate_data_0
for i in range(70):
    train_features = train_features.append(simulate_data_1)
train_labels = train_features.loc[:,'target']
del train_features['target']

#build mlp models
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128,activation='relu',kernel_initializer='uniform',input_dim=24))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu',kernel_initializer='uniform'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
model.load_weights('model.h5')

#training params
epochs = 50
batch_size = 128

#start training
train_history = model.fit(train_features,
                            train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            verbose=1)

print(train_history.history)
model.save('model.h5')
#predict values
test_labels = model.predict(test_fectures)
#save predictions
test_labels = pd.DataFrame(test_labels)
test_labels.rename(columns={0:"target"},inplace=True)
test_labels.index = range(132030,155590+1)
test_labels.index.name = 'id'
test_labels.to_csv(test_file_target)
print('done!')