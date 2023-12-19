from core.my_visual import gen_bar_plot

import pandas as pd
import models.lgbm as lgbm


if __name__ == '__main__':
    DEBUG = True
    dataset_path = 'dataset/tribunais_eleitorais/dataset_tre-ne.csv'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    importance_type = 'split'
    random_seed = 3
    top_n_import = 10
    splits_kfold = 10

    df = pd.read_csv(dataset_path, sep='\t')

    if DEBUG:
        print('#################################')
        print('#### Feature Importance LGBM ####')
        print('#################################\n')

    params = {}
    params['boosting_type'] = 'dart'
    params['learning_rate'] = 0.2
    params['n_estimators'] = 600

    if DEBUG:
        print('### Importance type: ' + importance_type)

    df_import = lgbm.get_feat_import_lgbm(df, 
                                          gt, 
                                          [gt], 
                                          random_seed, 
                                          importance_type, 
                                          params)

    gen_bar_plot(df_import, top_n_import)


    importance_type = 'gain'

    if DEBUG:
        print('### Importance type: ' + importance_type)

    df_import = lgbm.get_feat_import_lgbm(df, 
                                          gt, 
                                          [gt], 
                                          random_seed, 
                                          importance_type, 
                                          params)

    gen_bar_plot(df_import, top_n_import)

    print('done!')
    