# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:53:41 2020

@author: Admin
"""

import pandas as pd
import numpy as np
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


# =============================================================================
# Data import
# =============================================================================
credit_df=pd.read_csv("Credit_default_dataset.csv")
credit_df.head(6)

#drop the id column from dataframe since its a uniques identifire which is not require for prediction
credit_df.drop('ID', axis=1,inplace=True)

#changing the name of  pay_0 column to pay_1 to make the columne numbering correct
credit_df.rename(columns={"PAY_O":"PAY_1"},inplace=True)


# =============================================================================
# Data preprocessing steps
# =============================================================================
#Removing Unwanted categorical levels as mentioned in data exploration
credit_df['EDUCATION'].value_counts()
credit_df["EDUCATION"]=credit_df["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
credit_df["MARRIAGE"]=credit_df["MARRIAGE"].map({0:3,1:1,2:2,3:3})

from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
X=credit_df.drop(['default.payment.next.month'],axis=1)
X=scaling.fit_transform(X)

y=credit_df['default.payment.next.month']

## Hyper Parameter Optimization
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

# =============================================================================
# Hyperparameter optimization using RandomizedSearchCV
# =============================================================================
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

from datetime import datetime
start_time = timer(None) 
random_search.fit(X,y)
timer(start_time) 

#check the best parameter and estimator
random_search.best_estimator_
random_search.best_params_

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)
print(score)
print(score.mean())