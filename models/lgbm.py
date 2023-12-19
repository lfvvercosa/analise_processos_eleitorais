from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

import pandas as pd

DEBUG = True

def run_lgbm(df, gt, not_a_feature, params, random_seed, splits):
    
    if params is None:
        boost_type = ['gbdt', 'dart']
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6]
        n_estim = [100, 200, 400, 600, 800, 1000, 1200, 1600]
    else:
        boost_type = [params['boosting_type']]
        learn_rate = [params['learning_rate']]
        n_estim = [params['n_estimators']]

    min_mse = float('inf')
    min_mae = float('inf')
    min_r2 = float('inf')

    best_params = {}
    feat = [f for f in df.columns if f not in not_a_feature]
    
    for boost in boost_type:
        print('boost type: ' + str(boost))
        for learn in learn_rate:
            for n in n_estim:

                X = df[feat].to_numpy()
                y = df[gt].to_numpy()
                skf = KFold(n_splits=splits, 
                            shuffle=True, 
                            random_state=random_seed)
                skf.get_n_splits(X, y)
                
                mse = []
                r2 = []
                mae = []

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    reg = LGBMRegressor(boosting_type=boost,
                                        learning_rate=learn,
                                        n_estimators=n)
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    y_pred = [y if y > 0 else 0 for y in y_pred]

                    mse.append(mean_squared_log_error(y_test, y_pred))
                    r2.append(r2_score(y_test, y_pred))
                    mae.append(mean_absolute_error(y_test, y_pred))

                mse_mean = sum(mse) / len(mse)
                r2_mean = sum(r2) / len(r2)
                mae_mean = sum(mae) / len(mae)

                mse_var = sum([(x - mse_mean) ** 2 for x in mse]) / len(mse)
                r2_var = sum([(x - r2_mean) ** 2 for x in r2]) / len(mse)
                mae_var = sum([(x - mae_mean) ** 2 for x in mae]) / len(mse)

                mse_std = mse_var**0.5
                r2_std = r2_var**0.5
                mae_std = mae_var**0.5


                if mse_mean < min_mse:
                    print('#### MIN MSE LGBM: ' + str(mse_mean))
                    
                    min_mse = mse_mean
                    min_mae = mae_mean
                    min_r2 = r2_mean

                    min_mse_std = mse_std
                    min_mae_std = mae_std
                    min_r2_std = r2_std

                    best_params['boosting_type'] = boost
                    best_params['learning_rate'] = learn
                    best_params['n_estimators'] = n

    print('### best params LGBM: ' + str(best_params) + '\n')
    
    print('### best mse: ', min_mse)
    print('### best mse std: ' + str(min_mse_std) + '\n')

    print('### best mae: ', min_mae)
    print('### best mae std: ' + str(min_mae_std) + '\n')

    print('### best r2: ', min_r2)
    print('### best r2 std: ' + str(min_r2_std) + '\n')


    return best_params


def use_lgbm(X_train, y_train, X_test, best_params=None):
    if not best_params:
        reg = LGBMRegressor()
    else:
        
        boost = best_params['boosting_type'] 
        learn = best_params['learning_rate'] 
        n = best_params['n_estimators'] 

        reg = LGBMRegressor(boosting_type=boost,
                            learning_rate=learn,
                            n_estimators=n)

        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        y_pred = list(y_pred)
        y_pred = [x if x > 0 else 0 for x in y_pred]

    return y_pred


def get_feat_import_lgbm(df, 
                         gt, 
                         not_a_feature,
                         random_seed, 
                         importance_type, 
                         params):
    
    feat = [f for f in df.columns if f not in not_a_feature]
    X = df[feat].to_numpy()
    y = df[gt].to_numpy()
    splits = 10
    count = 0
    feature_names = df[feat].columns
    df_import = pd.DataFrame({'Feature': feature_names})

    skf = KFold(n_splits=splits, 
                shuffle=True, 
                random_state=random_seed)
    
    if DEBUG:
        print('Running LGBM...')

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg = LGBMRegressor(boosting_type=params['boosting_type'],
                            learning_rate=params['learning_rate'],
                            n_estimators=params['n_estimators'],
                            importance_type=importance_type)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_pred = [y if y > 0 else 0 for y in y_pred]

        

        df_import[str(count)] = reg.feature_importances_

        count += 1

    df_import = df_import.set_index('Feature')
    se_import = df_import.mean(axis=1)
    se_import = se_import.sort_values(ascending=False)


    return se_import