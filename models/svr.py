from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


def run_svr(df, gt, not_a_feature, params, random_seed):

    if params is None:
        C = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        epsilon = [0.1]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        tol = [0.001, 0.01, 0.1]
        
    else:
        C = [params['C']]
        epsilon = [params['epsilon']]
        kernel = [params['kernel']]
        tol = [params['tol']]

    best_params = {}

    min_mse = float('inf')
    min_mae = float('inf')
    min_r2 = float('inf')

    splits = 10
    feat = [f for f in df.columns if f not in not_a_feature]
        
    for curr_c in C:
        for eps in epsilon:
            for ker in kernel:
                for to in tol:

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

                            reg = SVR(kernel=ker,
                                      epsilon=eps,
                                      tol=to,
                                      C=curr_c)
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

                            min_mse = mse_mean
                            min_mae = mae_mean
                            min_r2 = r2_mean

                            min_mse_std = mse_std
                            min_mae_std = mae_std
                            min_r2_std = r2_std

                            print('#### MIN MSE SVR: ' + str(min_mse))
                            best_params['C'] = curr_c
                            best_params['epsilon'] = eps
                            best_params['kernel'] = ker
                            best_params['tol'] = to

    print('### BEST_PARAMS SVR: ' +str(best_params))

    print('### best mse: ', min_mse)
    print('### best mse std: ' + str(min_mse_std) + '\n')

    print('### best mae: ', min_mae)
    print('### best mae std: ' + str(min_mae_std) + '\n')

    print('### best r2: ', min_r2)
    print('### best r2 std: ' + str(min_r2_std) + '\n')

    
        
        

    return best_params


def use_svr(X_train, y_train, X_test, best_params=None):
    if not best_params:
        reg = SVR()
    else:
        reg = SVR(kernel=best_params['kernel'],
                  epsilon=best_params['epsilon'],
                  gamma=best_params['gamma'],
                  C=best_params['C'])

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_pred = list(y_pred)
    y_pred = [x if x > 0 else 0 for x in y_pred]

    return y_pred