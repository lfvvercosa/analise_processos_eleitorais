from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd

import core.my_loader as my_loader
from core import my_create_features


if __name__ == "__main__": 
    base_path = 'dataset/'
    log_path = 'dataset/tribunais_eleitorais/tre-ne.xes'
    congest_path = 'dataset/proc_taxa_congestionamento_ujs.csv'
    pend_path = 'dataset/proc_pendentes_serie_ujs.csv'
    out_path = 'dataset/tribunais_eleitorais/dataset_tre-ne.csv'
    DEBUG = True

    df_congest = pd.read_csv(congest_path, sep='\t')
    df_pend = pd.read_csv(pend_path, sep='\t')
    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    df_log = convert_to_dataframe(log)
    df_code_subj = my_loader.load_df_subject(base_path)
    df_code_mov = my_loader.load_df_movements(base_path)


    # Create feature dataset
    df_feat = my_create_features.create_features(df_log,
                                                 df_code_subj,
                                                 df_code_mov,
                                                 df_pend,
                                                 df_congest,
                                                 base_path)

    # Prepair feature dataset to the model
    df_feat = my_create_features.process_features(df_feat, base_path)

    # Save it
    df_feat.to_csv(out_path, sep='\t', index=False)

    if DEBUG:
        print('dataset created!')

