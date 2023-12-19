from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


def run_linear_regression(df, n_splits):

    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    feat = [c for c in df.columns if c != gt]

    X = df[feat].to_numpy()
    y = df[gt].to_numpy()

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    skf.get_n_splits(X, y)

    msle = []
    r2 = []
    mae = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        y_pred = linear_model.predict(X_test)

        y_pred = [y if y > 0 else 0 for y in y_pred]

        msle.append(mean_squared_log_error(y_test, y_pred))
        r2.append(r2_score(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))

    msle_mean = sum(msle) / len(msle)
    r2_mean = sum(r2) / len(r2)
    mae_mean = sum(mae) / len(mae)

    msle_var = sum([(x - msle_mean) ** 2 for x in msle]) / len(msle)
    r2_var = sum([(x - r2_mean) ** 2 for x in r2]) / len(msle)
    mae_var = sum([(x - mae_mean) ** 2 for x in mae]) / len(msle)

    mse_std = msle_var**0.5
    r2_std = r2_var**0.5
    mae_std = mae_var**0.5

    print('### lin reg mse: ', msle_mean)
    print('### lin reg mse std: ' + str(mse_std) + '\n')

    print('### lin reg mae: ', mae_mean)
    print('### lin reg mae std: ' + str(mae_std) + '\n')

    print('### lin reg r2: ', r2_mean)
    print('### lin reg r2 std: ' + str(r2_std) + '\n')
