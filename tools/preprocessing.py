# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:33:47 2021

@author: Administrator
处理数据流程：
->删除异常数据（1.异常字段，包括test与train分布差很多的；2.训练集中的离群数据）
->
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#
def showProbabilyPlot(col_target):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.distplot(col_target.dropna(),fit=stats.norm);
    plt.subplot(1,2,2)
    _=stats.probplot(col_target.dropna(), plot=plt)

"""
查看数据（训练集和测试集）分布
"""
def showDistribution(df_train,df_test,columns=[]):
    if len(columns) == 0:
        columns = df_train.columns
    for column in columns:
        g = sns.kdeplot(df_train[column], color="Red", shade = True)
        g = sns.kdeplot(df_test[column], ax =g, color="Blue", shade= True)
        g.set_xlabel(column)
        g.set_ylabel("Frequency")
        g = g.legend(["train","test"])
        plt.show()
        
def showCorrelation(df):
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);


# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)    
    n = len(y_pred)   
    
def find_outliers(model, X, y, sigma=3,show_info=False):

    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X,y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        
    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid)/std_resid    
    outliers = z[abs(z)>sigma].index
    
    if show_info:
        # print and plot the results
        print('R2=',model.score(X,y))
        print('rmse=',rmse(y, y_pred))
        print("mse=",mean_squared_error(y,y_pred))
        print('---------------------------------------')
    
        print('mean of residuals:',mean_resid)
        print('std of residuals:',std_resid)
        print('---------------------------------------')
    
        print(len(outliers),'outliers:')
        print(outliers.tolist())

    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,'.')
    plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],'ro')
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    z.loc[outliers].plot.hist(color='r',bins=50,ax=ax_133)
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('z')
    
    # plt.savefig('outliers.png')
    
    return outliers

#删除离群数据
def removeOutliers(X, y):
    outliers = find_outliers(Ridge(), X, y)
    return X.drop(outliers),y.drop(outliers)

#删除低相关性特征
def removeLowCorrelation(df_train,df_test,target,threshold = 0.1,concat=False):
    
    corr_matrix = df_train.corr().abs()
    drop_col=corr_matrix[corr_matrix[target] < threshold].index
    df_train.drop(drop_col,axis=1,inplace=True)
    df_test.drop(drop_col,axis=1,inplace=True)
    if concat:
        return pd.concat([df_train,df_test])
    else:
        return df_train,df_test

    
def get_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    return missing_data

def showNanData(df):
    np.isnan(df).any()

#min-max标准化    
def minmax_scaler(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

#one-hot编码
def onehot_encode(df,cols):
    for col in cols:
        for_dummy = df.pop(col)
        extra_data = pd.get_dummies(for_dummy,prefix=col)
        df = pd.concat([df, extra_data],axis=1)
    return df

def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())    
