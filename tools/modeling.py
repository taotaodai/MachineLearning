# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:26:16 2021

@author: Administrator
"""
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import xgboost
from xgboost import XGBRegressor
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


seed = 2018
def build_model():
    svr = make_pipeline(SVR(kernel='linear'))
    line = make_pipeline(LinearRegression())
    lasso = make_pipeline(Lasso(alpha =0.0005, random_state=1))
    ENet = make_pipeline(ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)    
    lgbm = lightgbm.LGBMRegressor(learning_rate=0.01, n_estimators=500, num_leaves=31)
    xgb = xgboost.XGBRegressor(booster='gbtree',colsample_bytree=0.8, gamma=0.1, 
                                 learning_rate=0.02, max_depth=5, 
                                 n_estimators=500,min_child_weight=0.8,
                                 reg_alpha=0, reg_lambda=1,
                                 subsample=0.8, silent=1,
                                 random_state =seed, nthread = 2)
    nn = KerasRegressor(build_fn=build_nn, nb_epoch=500, batch_size=32, verbose=2)
    return svr, line, lasso, ENet, KRR1, KRR2, lgbm, xgb, nn

def build_nn():
    model = Sequential()
    model.add(Dense(units=128, activation='linear', input_dim=18))
    model.add(Dense(units=32, activation='linear'))
    model.add(Dense(units=8, activation='linear'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')