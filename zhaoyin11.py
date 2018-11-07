# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:35:55 2018

@author: YUBO
"""

import pandas as pd
from collections import Counter
data=pd.read_csv('train_agg.csv')
data.apply(lambda x:x.replace('"',''))
data.to_csv('data.csv',index=False)
del data
train_agg=pd.read_csv('data.csv',delimiter='\t')
train_flg=pd.read_csv('train_flg.csv',delimiter='\t')
train=pd.merge(train_agg,train_flg,on='USRID')
train_log=pd.read_csv('train_log.csv',delimiter='\t')
test_agg=pd.read_csv('test_agg.csv',delimiter='\t')
test_log=pd.read_csv('test_log.csv',delimiter='\t')
def data_process(data):
    data.TCH_TYP.replace(0,'APP',inplace=True)
    data.TCH_TYP.replace(2,'H5',inplace=True)
    
def cut1(group):
    return group.split('-')[0]
def cut2(group):
    return group.split('-')[1]
def cut3(group):
    return group.split('-')[2]

t1=train_log['EVT_LBL'].apply(cut1)
t2=train_log['EVT_LBL'].apply(cut2)
t3=train_log['EVT_LBL'].apply(cut3)
train_log['EVT_LBL1']=t1
train_log['EVT_LBL2']=t2
train_log['EVT_LBL3']=t3

#分析每个用户对应每个界面的次数
m1=[]
m2=[]
m3=[]
gp1=train_log.groupby('USRID').agg({'EVT_LBL1':'unique'}).reset_index()
for i in range(0,len(gp1)):
    m1.append(len(gp1['EVT_LBL1'][i]))
gp1['count1']=m1

def data_log_process(data_log):#每天浏览时次数，每月浏览次数，模块浏览次数，浏览频繁时间段，每天浏览间隔，
    data_log['hour']=pd.to_datetime(data_log.OCC_TIM).dt.hour
    data_log['day']=pd.to_datetime(data_log.OCC_TIM).dt.day
    data_log['minute']=pd.to_datetime(data_log.OCC_TIM).dt.minute
    data_log['second']=pd.to_datetime(data_log.OCC_TIM).dt.second
data_log_process(train_log)
ss=train_log.groupby(by=['USRID','day'])[['minute']].count().reset_index()
ss=ss.groupby('USRID').apply(lambda t: t[t.minute==t.minute.max()])
ss2=ss.groupby('USRID').agg({'day':'min'})

l=[]
for t1 in ss2['day']:
    l.append(t1)
gp1['time']=l

gp2=train_log.groupby('USRID').agg({'EVT_LBL2':'unique'}).reset_index()
for i in range(0,len(gp2)):
    m2.append(len(gp2['EVT_LBL2'][i]))
gp2['count2']=m2

gp3=train_log.groupby('USRID').agg({'EVT_LBL3':'unique'}).reset_index()
for i in range(0,len(gp3)):
    m3.append(len(gp3['EVT_LBL3'][i]))
gp3['count3']=m3
#统计总次数
gp4=train_log.groupby('USRID').agg({'EVT_LBL3':'count'}).reset_index()
#
index=train_agg['USRID']
index=pd.DataFrame(index)
index=index.merge(gp1, on=['USRID'], how='left')
index=index.merge(gp2, on=['USRID'], how='left')
index=index.merge(gp3, on=['USRID'], how='left')
index=index.merge(gp4, on=['USRID'], how='left')
index.drop('EVT_LBL1',axis=1, inplace=True)
index.drop('EVT_LBL2',axis=1, inplace=True)
index.drop('EVT_LBL3_x',axis=1, inplace=True)
index.rename(columns={'EVT_LBL3_y':'total'}, inplace = True)


 
