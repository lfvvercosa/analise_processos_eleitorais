from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from core.my_visual import gen_bar_plot
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from scipy import stats
import pandas as pd
import numpy as np

import models.svr as svr
import models.lgbm as lgbm
import models.naive as naive
import models.linear as linear

import boto3


if __name__ == '__main__':
    DEBUG = True
    dataset_path = 'dataset/tribunais_eleitorais/dataset_tre-ne.csv'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    random_seed = 3
    splits_kfold = 10


    df = pd.read_csv(dataset_path, sep='\t')
    

    if DEBUG:
        print('#####################')
        print('#### apply Naive ####')
        print('#####################\n')

    naive.run_naive(df, splits_kfold)    

    if DEBUG:
        print('#############################')
        print('## apply Linear Regression ##')
        print('#############################\n')

    linear.run_linear_regression(df, splits_kfold)


    if DEBUG:
        print('#####################')
        print('#### apply LGBM ####')
        print('#####################\n')

    params = {}
    params['boosting_type'] = 'dart'
    params['learning_rate'] = 0.2
    params['n_estimators'] = 600

    best_params_model = lgbm.run_lgbm(df, 
                                      gt, 
                                      [gt], 
                                      params, 
                                      random_seed, 
                                      splits_kfold)
    
    if DEBUG:
        print('#####################')
        print('##### apply SVR #####')
        print('#####################\n')

    params = {}
    params['C'] = 1024
    params['kernel'] = 'rbf'
    params['gamma'] = 'scale'
    params['epsilon'] = 0.1
    params['tol'] = 0.001

    best_params_model = svr.run_svr(df, gt, [gt], params, random_seed)