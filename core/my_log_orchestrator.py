import pandas as pd
import os
import numpy as np
import pm4py
import pickle
import matplotlib.pyplot as plt
from collections import Counter

import core.my_loader as my_loader
import core.my_filter as my_filter
import core.my_parser as my_parser
import core.my_utils as my_utils
import core.my_stats as my_stats


def load_dataframes(my_justice, just_specs, base_path):
    df = my_loader.load_just_spec_df(just_specs, 
                                     base_path, 
                                     my_justice)
    df = my_parser.parse_data(df)

    df_assu = df[['id',
                  'processoNumero',
                  'assuntoCodigoNacional',
                  'classeProcessual']].\
            explode('assuntoCodigoNacional')
    df_assu = df_assu[~df_assu['assuntoCodigoNacional'].isna()]

    df_mov = df.explode('movimento')
    df_mov = my_parser.parse_data_mov(df_mov)


    return df, df_assu, df_mov


def pre_process( 
                df_assu, 
                df_mov,
                df_code_subj,
                df_code_type,
                df_code_mov,
                start_limit,
                time_outlier
                ):
    
    ## Removing null traces

    print('### Total traces start:')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = my_filter.filter_null_traces(df_mov)

    print('### Total traces after removing null')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Removing invalid traces

    df_mov = my_filter.filter_invalid_traces(df_mov,
                                            df_assu,
                                            df_code_subj,
                                            df_code_type,
                                            df_code_mov)

    print('### Total traces after removing invalid')
    print(my_utils.get_total_traces_df_mov(df_mov))
    
    ## Remove traces that contain movement after year limit

    df_temp = df_mov.copy()
    df_temp['YEAR'] = df_temp['movimentoDataHora'].dt.year
    df_temp = df_temp[df_temp['YEAR'] > start_limit]
    df_mov = df_mov[~df_mov['id'].isin(df_temp['id'])]

    print('### Total traces after limiting movement year')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Map movements

    level_magistrado = 1
    level_serventuario = 2

    df_mov = my_utils.map_movements(df_mov, 
                                    df_code_mov, 
                                    level_magistrado,
                                    level_serventuario)

    ## Map Type attribute

    level = 1
    df_mov = my_utils.map_type(df_mov,
                               df_code_type,
                               level)


    print('### Total traces after mapping movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = df_mov.sort_values(['id','movimentoDataHora'])
    df_mov = df_mov.reset_index(drop=True)

    ## Remove traces with single activity

    print('### Total traces before filtering single movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = my_filter.filter_single_act_traces(df_mov)

    print('### Total traces after filtering single movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Remove repeated activities in sequence (autoloops)

    # df_mov = my_filter.rem_autoloop2(df_mov, 
    #                                 id_col='id', 
    #                                 act_col='movimentoCodigoNacional')


    ## Remove traces with rare start or end activities 
    ## (possibly didnt yet finish or had already started)

    # print('### Total traces after removing specific activities' + \
    #       ' and autoloop')
    # print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = my_filter.filter_non_complete_traces(df_mov, 0.03)

    print('### Total traces after filtering non-complete')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Remove traces with outlier duration

    df_dur = my_stats.get_traces_duration(df_mov)

    # df_dur.hist()
    # plt.show()

    min_val = time_outlier
    max_val = 1 - time_outlier
    df_mov = my_filter.filter_outlier_duration_trace(df_mov, min_val, max_val)

    print('### Total traces after filtering outlier duration')
    print(my_utils.get_total_traces_df_mov(df_mov))

    # Check traces by starting year
    df_work = df_mov[['id','movimentoDataHora']]
    df_work['year'] = df_work['movimentoDataHora'].dt.year
    df_start = df_work.groupby('id').agg(start_year=('year','min'))
    df_start['start_year'].hist()

    return df_mov


def create_xes_file(df_mov, path):
    case_id = 'id'
    activity = 'movimentoCodigoNacional'
    timestamp = 'movimentoDataHora'

    df_mov = df_mov.rename(columns={
        case_id:'case:concept:name',
        activity:'concept:name',
        timestamp:'time:timestamp',
        'classeProcessual':'case:lawsuit:type',
        'assuntoCodigoNacional':'case:lawsuit:subjects',
        'processoNumero':'case:lawsuit:number',
        'orgaoJulgadorCodigo':'case:court:code',
        'processoEletronico':'case:digital_lawsuit',
        'nivelSigilo':'case:secrecy_level',
    })

    df_mov['case:concept:name'] = df_mov['case:concept:name'].astype(str)

    log = pm4py.convert_to_event_log(df_mov)
    pm4py.write_xes(log, path)

    print('xes file "' + path +  '" was created.')