# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:35:55 2018

@author: YUBO
"""

import pandas as pd
from pandas import DataFrame
from collections import Counter
from scipy.sparse.construct import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder
import numpy as np
from sklearn import metrics
data=pd.read_csv('train_agg.csv')
data.apply(lambda x:x.replace('"',''))
data.to_csv('data.csv',index=False)
del data
train_agg=pd.read_csv('data.csv',delimiter='\t')
train_flg=pd.read_csv('train_flg.csv',delimiter='\t')
train_agg=pd.merge(train_agg,train_flg,on='USRID')
train_log=pd.read_csv('train_log.csv',delimiter='\t')
test_agg=pd.read_csv('test_agg.csv',delimiter='\t')
test_log=pd.read_csv('test_log.csv',delimiter='\t')


len_train=len(train_agg)
merge_log=train_log.append(test_log)

def data_process(data):
    data.TCH_TYP.replace(2,1,inplace=True)
data_process(merge_log)   
def cut1(group):
    return group.split('-')[0]
def cut2(group):
    return group.split('-')[1]
def cut3(group):
    return group.split('-')[2]

t1=merge_log['EVT_LBL'].apply(cut1)
t2=merge_log['EVT_LBL'].apply(cut2)
t3=merge_log['EVT_LBL'].apply(cut3)
merge_log['EVT_LBL1']=t1
merge_log['EVT_LBL2']=t2
merge_log['EVT_LBL3']=t3

#分析每个用户对应每个界面的次数
m1=[]
m2=[]
m3=[]
gp1=merge_log.groupby('USRID').agg({'EVT_LBL1':'unique'}).reset_index()
for i in range(0,len(gp1)):
    m1.append(len(gp1['EVT_LBL1'][i]))
gp1['count1']=m1

def data_log_process(data_log):#每天浏览时次数，每月浏览次数，模块浏览次数，浏览频繁时间段，每天浏览间隔，
    data_log['hour']=pd.to_datetime(data_log.OCC_TIM).dt.hour
    data_log['day']=pd.to_datetime(data_log.OCC_TIM).dt.day
    data_log['minute']=pd.to_datetime(data_log.OCC_TIM).dt.minute
    
data_log_process(merge_log)
import time,datetime
def Caltime(date):
    date1=time.strptime(min(date),"%Y-%m-%d %H:%M:%S")
    date2=time.strptime(max(date),"%Y-%m-%d %H:%M:%S")
    date1=datetime.datetime(date1[0],date1[1],date1[2],date1[3],date1[4],date1[5])
    date2=datetime.datetime(date2[0],date2[1],date2[2],date2[3],date2[4],date2[5])
    return (date2-date1).days

gp=merge_log.groupby(by=['USRID']).agg({'OCC_TIM':Caltime}).reset_index().rename(index=str, columns={'OCC_TIM': 'DIF'})
gp_day_mean=merge_log.groupby(by=['USRID']).agg({'day':'mean'}).reset_index().rename(index=str, columns={'day': 'day_mean'})
gp_day_var=merge_log.groupby(by=['USRID']).agg({'day':'var'}).reset_index().rename(index=str, columns={'day': 'day_var'})
ss=merge_log.groupby(by=['USRID','day'])[['minute']].count().reset_index()
a=ss.groupby(by=['USRID'])[['day']].count().reset_index()
ss=ss.merge(a,on='USRID',how='left')
ss.rename(index=str,columns={'day_x':'day'},inplace=True)
ss=ss.groupby('USRID').apply(lambda t: t[t.minute==t.minute.max()])

ss2=ss.groupby('USRID').agg({'day':'min'})

l=[]
for t1 in ss2['day']:
    l.append(t1)
gp1['time']=l

gp2=merge_log.groupby('USRID').agg({'EVT_LBL2':'unique'}).reset_index()
for i in range(0,len(gp2)):
    m2.append(len(gp2['EVT_LBL2'][i]))
gp2['count2']=m2

gp3=merge_log.groupby('USRID').agg({'EVT_LBL3':'unique'}).reset_index()
for i in range(0,len(gp3)):
    m3.append(len(gp3['EVT_LBL3'][i]))
gp3['count3']=m3
#统计总次数
gp4=merge_log.groupby('USRID').agg({'EVT_LBL3':'count'}).reset_index()
#
index1=train_agg['USRID'].append(test_agg['USRID'])
index1=pd.DataFrame(index1)
index1=index1.merge(gp, on=['USRID'], how='left')
index1=index1.merge(gp_day_mean, on=['USRID'], how='left')
index1=index1.merge(gp_day_var, on=['USRID'], how='left')
index1=index1.merge(gp1, on=['USRID'], how='left')
index1=index1.merge(gp2, on=['USRID'], how='left')
index1=index1.merge(gp3, on=['USRID'], how='left')
index1=index1.merge(gp4, on=['USRID'], how='left')
index1.drop('EVT_LBL1',axis=1, inplace=True)
index1.drop('EVT_LBL2',axis=1, inplace=True)
index1.drop('EVT_LBL3_x',axis=1, inplace=True)
index1.rename(columns={'EVT_LBL3_y':'total'}, inplace = True)
train_log=index1.ix[:(len_train-1),:]
test_log=index1.ix[len_train:,:]

cnt=[]
for col in train_agg.columns:
    x=len(train_agg[col].unique())
    cnt.append((col,x))
from sklearn.preprocessing import LabelEncoder
def data_process(data):
     encoder = LabelEncoder()
     data['V2'] = encoder.fit_transform(data['V2'])
     data['V4'] = encoder.fit_transform(data['V4'])
     data['V5'] = encoder.fit_transform(data['V5']) 
data_process(train_agg)
data_process(test_agg)

del a,gp,gp_day_mean,gp_day_var,gp1,gp2,gp3,gp4,index1,l,m1,m2,m3,merge_log,ss,ss2,t1,t2,t3,train_flg
#gbdt 构造新特征
gbdt = GradientBoostingClassifier(loss='exponential',learning_rate=0.12,n_estimators=60, max_depth=3,random_state=42,max_features=None)
X_train=train_agg.drop(['USRID','FLAG'],axis=1)
y_train=train_agg['FLAG']
# 训练学习
gbdt.fit(X_train, y_train)
# GBDT编码原有特征
X_train_leaves = gbdt.apply(X_train)[:,:,0]
X_test_leaves=gbdt.apply(test_agg.drop('USRID',axis=1))[:,:,0]
(train_rows, cols) = X_train_leaves.shape
onehot = OneHotEncoder()
X_trans = onehot.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

# 组合特征
X_train_agg = DataFrame(hstack([X_trans[:train_rows, :], train_agg]).toarray())
X_test_agg = DataFrame(hstack([X_trans[train_rows:, :], test_agg]).toarray())
X_train_agg.rename(columns={494: "USRID",495:"FLAG"},inplace=True)
X_test_agg.rename(columns={494: "USRID"},inplace=True)

#训练集和测试集

train_data=pd.merge(X_train_agg,train_log,on='USRID',how='left')
test_data=pd.merge(X_test_agg,test_log,on='USRID',how='left')
del X_train_agg,X_test_agg,train_log,test_log
#建模
import lightgbm as lgb
train_xy,offline_test = train_test_split(train_data,test_size = 0.3,random_state=42)
train,val = train_test_split(train_xy,test_size = 0.3,random_state=42)

# 训练集
y_train = train.FLAG                                              # 训练集标签
X_train = train.drop(['USRID','FLAG'],axis=1)                # 训练集特征矩阵
 
 # 验证集
y_val = val.FLAG                                                   # 验证集标签
X_val = val.drop(['USRID','FLAG'],axis=1)                    # 验证集特征矩阵
 
 # 测试集
offline_test_X = offline_test.drop(['USRID','FLAG'],axis=1)  # 线下测试特征矩阵
online_test_X  = test_data.drop(['USRID'],axis=1)              # 线上测试特征矩阵

### 数据转换
print('数据转换')
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)
 
### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {'is_unbalance': 'true',
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'tree_learner':'data',
          'learning_rate':0.2
          }

 ### 交叉验证(调参)
print('交叉验证')
min_auc = 0.8
best_params = {}

# 准确率
print("调参1：提高准确率")
for num_leaves in range(5,200,5):
    for max_depth in range(3,8,1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=2018,
                            nfold=3,
                            metrics=['auc'],
                            early_stopping_rounds=5,
                            verbose_eval=True, 
                            categorical_feature=['time']
                             )

        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
        if mean_auc > min_auc:
            
            min_auc = mean_auc
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
 
params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

 # 过拟合
print("调参2：降低过拟合")
for max_bin in range(1,255,5):
    
    for min_child_weight in range(1,100,5):
         
        params['max_bin'] = max_bin
        params['min_child_weight'] = min_child_weight
 
        cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=42,
                                nfold=3,
                                metrics=['auc'],
                                early_stopping_rounds=5,
                                verbose_eval=True,
                                categorical_feature=['time']
                                )
 
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
 
        if mean_auc > min_auc:
              
            min_auc = mean_auc
            best_params['max_bin']= max_bin
            best_params['min_child_weight'] = min_child_weight
 
params['min_child_weight'] = best_params['min_child_weight']
params['max_bin'] = best_params['max_bin']

print("调参3：降低过拟合")
for feature_fraction in [0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]:
    
    for bagging_fraction in [0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]:
        for bagging_freq in range(0,50,5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq

            cv_results = lgb.cv(params,
                                lgb_train,
                                seed=42,
                                nfold=3,
                                metrics=['auc'],
                                early_stopping_rounds=5,
                                verbose_eval=True,
                                categorical_feature=['time']
                                 )
 
            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
 
            if mean_auc > min_auc:
                
                min_auc = mean_auc
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq
params['feature_fraction'] = best_params['feature_fraction']
params['bagging_fraction'] = best_params['bagging_fraction']
params['bagging_freq'] = best_params['bagging_freq']
 
print("调参4：降低过拟合")
for lambda_l1 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,10,100]:
    for lambda_l2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,10,100]:
        for min_split_gain in [0,0.01,0.001,0.0001,0.1,0.2,0.3,0.4]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            params['min_split_gain'] = min_split_gain
 
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=42,
                                nfold=3,
                                metrics=['auc'],
                                early_stopping_rounds=5,
                                verbose_eval=True,
                                categorical_feature=['time']
                                 )

            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
 
            if mean_auc > min_auc:
                
                min_auc = mean_auc
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
                best_params['min_split_gain'] = min_split_gain
 
params['lambda_l1'] = best_params['lambda_l1']
params['lambda_l2'] = best_params['lambda_l2']
params['min_split_gain'] = best_params['min_split_gain'] 
print(best_params)
 
### 训练
params = {'is_unbalance': 'true',
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'tree_learner':'data',
          'learning_rate':0.1,
          'num_leaves': 10, 
           'max_depth': 3, 
           'max_bin': 1,
           'min_child_weight': 76, 
           'feature_fraction': 0.4, 
           'bagging_fraction': 0.8, 
           'bagging_freq': 5, 
           'lambda_l1': 10, 
           'lambda_l2': 100, 
           'min_split_gain': 0,
           'subsample_for_bin': 10000,
          #'scale_pos_weight':16
           'random_state':42
          }

params['learning_rate']=0.2

lgb_model=lgb.train(
           params,                     # 参数字典
           lgb_train,                  # 训练集
           valid_sets=lgb_eval,        # 验证集
           num_boost_round=2000,       # 迭代次数
           early_stopping_rounds=50,
           categorical_feature=['time']# 早停次数
          )
 
### 线下预测
print ("线下预测")
preds_offline = lgb_model.predict(offline_test_X, num_iteration=lgb_model.best_iteration) # 输出概率
offline=offline_test[['USRID','FLAG']]
offline['preds']=preds_offline
offline.FLAG = offline['FLAG'].astype(np.float64)
fpr, tpr, thresholds = metrics.roc_curve(offline.FLAG,offline['preds'])
print('AUC:',metrics.auc(fpr, tpr))

### 线上预测
print("线上预测")
online=DataFrame()
preds_online =  lgb_model.predict(online_test_X, num_iteration=lgb_model.best_iteration)  # 输出概率
online['USRID']=test_data['USRID'].astype(int)
online['RST']=preds_online
online.to_csv("test_result.csv",index=False,sep='\t')                   # 保存结果
