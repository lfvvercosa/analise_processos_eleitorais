from core import my_orchestrator
from core import my_loader

import unidecode
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler

DEBUG = True


def get_time_lawsuit(df):
    df_work = df.groupby('case:concept:name').\
                    agg(min_time=('time:timestamp','min'),
                        max_time=('time:timestamp','max'))
    
    df_work['total_time'] = (df_work['max_time'] - df_work['min_time']).dt.days
    df_work = df_work.drop(columns=['min_time','max_time'])


    return df_work


def total_distinct_subjects(subjects):
    l = eval(subjects)
    
    
    return len(set(l))


def extract_subject(subjects):
    l = eval(subjects)


    return l[0]


def get_subject_level(breadscrum, level):
    if type(breadscrum) == str:
        breadscrum = breadscrum.split(':')

        if len(breadscrum) > level:
            return int(breadscrum[level])
        else:
            return -1
    else:
        return -1
    

def map_subject(df, df_code_subj, level):
    temp = df_code_subj[['assuntoCodigoNacional','breadscrum']]
    temp = temp.rename(columns={'assuntoCodigoNacional':'case:lawsuit:subjects'})
    
    df = df.merge(temp, on=['case:lawsuit:subjects'], how='left')

    df['case:lawsuit:subjects'] = df.apply(lambda df: \
                                      get_subject_level(
                                        df['breadscrum'],
                                        level), axis=1)
    
    df_temp = df[df['case:lawsuit:subjects'] == -1]

    df = df[~df['case:concept:name'].isin(df_temp['case:concept:name'])]
    df = df.drop(columns=['breadscrum'])


    return df


def count_zeroes_col(df, thres):
    total = len(df.index)
    zeroes_perc = {}
    remove_cols = []

    cols = df.columns
    cols = [c for c in cols if c[-5:-2] == '-20']

    for c in cols:
        if c != 'case:court:code':
            zeroes = df[c].value_counts()[0]
            zeroes_perc[c] = round(zeroes/total,2)
            
            if zeroes_perc[c] > thres:
                remove_cols.append(c)
    

    return remove_cols


def rename_mov_cols(df, df_code_mov, ngram):
    df_temp = df_code_mov[['movimentoCodigoNacional','movimentoNome']]
    df_temp = df_temp.set_index('movimentoCodigoNacional')
    map_vals = df_temp.to_dict()['movimentoNome']
    rename_cols = {}

    for c in df.columns:
        try:
            if ngram == 1:
                name = map_vals[int(c)]
                name = name.replace(' ', '_').upper()
                name = name.replace('/', '_').upper()
                name = unidecode.unidecode(name)
                name = 'MOV_' + name

                rename_cols[c] = name
            if ngram == 2:
                t = eval(c)
                name1 = apply_to_mov_name(t[0], map_vals)
                name2 = apply_to_mov_name(t[1], map_vals)
                name = name1 + '=>' + name2

                rename_cols[c] = name
        except:
            pass

    df = df.rename(columns=rename_cols)


    return df


def rename_type(df, df_code_type):
    df_temp = df_code_type[['classeProcessual','classeNome']]
    df_temp = df_temp.rename(columns={'classeProcessual':'case:lawsuit:type'})

    df = df.merge(df_temp, on='case:lawsuit:type', how='left')
    df['case:lawsuit:type_temp'] = df['classeNome'] + '_' + \
        df['case:lawsuit:type'].astype(str)
    df['case:lawsuit:type_temp'] = 'CLA_' + df['case:lawsuit:type_temp'].\
        str.replace(' ', '_').str.upper()
    df['case:lawsuit:type_temp'] = df['case:lawsuit:type_temp'].apply(remove_accents)
    df = df.drop(columns=['case:lawsuit:type','classeNome'])
    df = df.rename(columns={'case:lawsuit:type_temp':'case:lawsuit:type'})
    
    return df


def rename_subject(df, df_code_subj):
    df_temp = df_code_subj[['assuntoCodigoNacional','assuntoNome']]
    df_temp = df_temp.rename(columns={'assuntoCodigoNacional':'case:lawsuit:subjects'})

    df = df.merge(df_temp, on='case:lawsuit:subjects', how='left')
    df['case:lawsuit:subjects_temp'] = df['assuntoNome'] + '_' + \
        df['case:lawsuit:subjects'].astype(str)
    df['case:lawsuit:subjects_temp'] = 'ASSU_' + df['case:lawsuit:subjects_temp'].\
        str.replace(' ', '_').str.upper()
    df['case:lawsuit:subjects_temp'] = df['case:lawsuit:subjects_temp'].apply(remove_accents)
    df = df.drop(columns=['case:lawsuit:subjects','assuntoNome'])
    df = df.rename(columns={'case:lawsuit:subjects_temp':'case:lawsuit:subjects'})
    
    return df


def rename_classific(df):
    df['Classificação da unidade'] = 'CLAS_' + df['Classificação da unidade'].\
        str.replace('-','_').str.replace(' ','_').str.replace(';','').\
        str.replace('(','').str.replace(')','').str.replace(',','').str.upper()
    df['Classificação da unidade'] = df['Classificação da unidade'].apply(remove_accents)

    return df


def remove_accents(name):
    return unidecode.unidecode(name)


def standard_name_cols(df, cols):
    std_name = {}

    for c in df.columns:
        if c not in cols:
            name = c.replace(' ','_').replace('-','_').upper()
            name = unidecode.unidecode(name)
            std_name[c] = name

    df = df.rename(columns=std_name)


    return df


def apply_to_mov_name(name, map_vals):
    name = map_vals[int(name)]
    name = name.replace(' ', '_').upper()
    name = name.replace('/', '_').upper()
    name = unidecode.unidecode(name)
    name = 'MOV_' + name


    return name


def group_infrequent_categoric(df, col, thres):
    df_temp = df.groupby(col).agg(count=(col,'count'))
    total = df_temp.sum()[0]
    min_val = thres * total
    updated_vals = {}
    curr_vals = df_temp.to_dict()['count']


    for k in curr_vals:
        if curr_vals[k] > min_val:
            updated_vals[k] = k
        else:
            updated_vals[k] = 'CLAS_OUTRO_' + col
    
    classe_processual = [k for k in updated_vals]
    classe_processual_mapped = [v for v in updated_vals.values()]
    
    df_map = pd.DataFrame.from_dict({
        col:classe_processual,
        col+'_MAPEADA':classe_processual_mapped,
        })
    
    df = df.merge(df_map, on=col, how='left')
    df = df.drop(columns=col)
    df = df.rename(columns={col+'_MAPEADA':col})

    return df


def group_infrequent_tipo(df):
    df['TIPO'] = df['TIPO'].replace('AADJ','Desconhecido')
    df['TIPO'] = df['TIPO'].replace('-','Desconhecido')
    df['TIPO'] = df['TIPO'].replace('Desconhecido','TIPO_DESCONHECIDO')
    df['TIPO'] = df['TIPO'].replace('UJ1','TIPO_UJ1')
    df['TIPO'] = df['TIPO'].replace('UJ2','TIPO_UJ2')


    return df


def winsorize_col(s, min_perc, max_perc):
    temp = winsorize(s, (min_perc,max_perc))
        
    if temp.min() != temp.max():
        s = temp

    return s


def convert_categoric_one_hot_encoder(df, col):
    # Get one hot encoding of column
    one_hot = pd.get_dummies(df[col])
    # Drop column as it is now encoded
    df = df.drop(col, axis = 1)
    # Join the encoded df
    df = df.join(one_hot)
    
    
    return df


def normalize_cols(df, id_col):
    cols_gt = ['TEMPO_PROCESSUAL_TOTAL_DIAS',
               'TEMPO_PROCESSUAL_TOTAL_CLASSES']
    scaler = MinMaxScaler()
    cols_norm = [c for c in df.columns if c not in cols_gt and c not in id_col]

    df[cols_norm] = scaler.fit_transform(df[cols_norm])
    

    return df


def create_features(df,
                    df_code_subj,
                    df_code_mov,
                    df_pend,
                    df_congest,
                    base_path,
                    ):
    level_subject = 1
    n = 1
    min_perc = 0.05
    max_perc = 0.95

    if DEBUG:
        print('total lines: ' + str(len(df.index)))

    ### Get time lawsuit ###
    df_time = get_time_lawsuit(df)

    ### Handle lawsuit subject feature ###
    if DEBUG:
        print('extract subject...')

    df['total_distinct_subjects'] = df.apply(lambda df: total_distinct_subjects(
                                                df['case:lawsuit:subjects']), axis=1
                                            )
    df['case:lawsuit:subjects'] = df.apply(lambda df: extract_subject(
                                                df['case:lawsuit:subjects']), axis=1
                                            )
    if DEBUG:
        print('map subject...')

    df = map_subject(df,
                     df_code_subj,
                     level_subject)
    
    ### Create 1-gram ###
    df_gram = my_orchestrator.create_1_gram_features(df, min_perc, max_perc, n)

    ### Create total movements features
    df_total_movs = df.groupby('case:concept:name').\
                       agg(movements_count=('concept:name','count'))
    df_total_movs_first_level = my_orchestrator.\
                                    get_total_movs_first_level(df, df_code_mov)
    
    ### Merge dataframes
    df_gram = df_gram.merge(df_time, on='case:concept:name', how='left')
    df_gram = df_gram.merge(df_total_movs, on='case:concept:name', how='left')
    df_gram = df_gram.merge(df_total_movs_first_level, 
                            on='case:concept:name', 
                            how='left')

    df_merge = df[[
        'case:concept:name',
        'case:lawsuit:type',
        'case:lawsuit:number',
        'case:court:code',
        'case:digital_lawsuit',
        'case:lawsuit:subjects',	
        'case:secrecy_level',
        'total_distinct_subjects',
    ]]

    df_merge = df_merge.drop_duplicates(subset='case:concept:name')
    df_merge = df_merge.set_index('case:concept:name')

    df_gram = df_gram.merge(df_merge, on='case:concept:name', how='left')
    df_gram = df_gram.reset_index()

    df_gram = df_gram.merge(df_pend, on='case:court:code', how='left')
    df_gram = df_gram.merge(df_congest, on='case:court:code', how='left')

    
    return df_gram


def process_features(df_feat, base_path):
    thres = 0.6
    ngram = 1
    id_col = ['case:concept:name']

    # Remove repeated or not useful features
    df_feat = df_feat.drop(columns=[
        'Classificação',
        'Tipo de unidade',
        'Unidade Judiciária',
        'Município',
        'Município sede',
        'Justiça',
        'Tribunal',
        'case:concept:name',
        'case:lawsuit:number',
        'case:secrecy_level',
    ])

    ## Fill nulls
    df_feat['Tipo'] = df_feat['Tipo'].fillna('Desconhecido')
    df_feat['Classificação da unidade'] = \
        df_feat['Classificação da unidade'].fillna('Desconhecido')
    
    # Remove columns with high null (or zero) percentage (above 60%)    
    rem_cols = count_zeroes_col(df_feat, thres)
    df_feat = df_feat.drop(columns=rem_cols)

    # Rename columns
    df_code_mov = my_loader.load_df_movements(base_path)
    df_code_type = my_loader.load_df_classes(base_path)
    df_code_subj = my_loader.load_df_subject(base_path)
    
    df_feat = rename_mov_cols(df_feat, df_code_mov, ngram)
    df_feat = rename_type(df_feat, df_code_type)
    df_feat = rename_subject(df_feat, df_code_subj)
    df_feat = rename_classific(df_feat)

    rename = {
        'case:lawsuit:subjects':'ASSUNTO_PROCESSUAL',
        'case:lawsuit:type':'CLASSE_PROCESSUAL',
        'case:digital_lawsuit':'PROCESSO_DIGITAL',
        'total_distinct_subjects':'TOTAL_ASSUNTOS',
        'total_time':'TEMPO_PROCESSUAL_TOTAL_DIAS',
    }

    df_feat = df_feat.rename(columns=rename)
    df_feat = standard_name_cols(df_feat, id_col)

    df_feat = group_infrequent_tipo(df_feat)
    df_feat = group_infrequent_categoric(df_feat, 'CLASSE_PROCESSUAL', 0.02)
    df_feat = group_infrequent_categoric(df_feat, 'ASSUNTO_PROCESSUAL', 0.02)
    df_feat = group_infrequent_categoric(df_feat, 'CLASSIFICACAO_DA_UNIDADE', 0.01)

    # Winsorizing outliers from numeric cols
    categoric_boll_cols_and_target = [
        'CLASSE_PROCESSUAL',
        'ASSUNTO_PROCESSUAL',
        'CLASSIFICACAO_DA_UNIDADE',
        'UF',
        'TIPO',
        'TEMPO_PROCESSUAL_TOTAL_DIAS',
        'PROCESSO_DIGITAL',
        'CASE:COURT:CODE',
    ]

    df_feat['CASE:COURT:CODE'] = df_feat['CASE:COURT:CODE'].astype(int)


    min_perc = 0.05
    max_perc = 0.05

    if DEBUG:
        print('Winsorizing outliers for columns:\n')

    for c in df_feat.columns:
        if c not in categoric_boll_cols_and_target and c not in id_col:
            if DEBUG:
                print(c)
            df_feat[c] = winsorize_col(df_feat[c], 
                                        min_perc, 
                                        max_perc)

    # Convert categoric to numeric
    df_feat = convert_categoric_one_hot_encoder(df_feat, 'TIPO')
    df_feat = convert_categoric_one_hot_encoder(df_feat, 'UF')
    df_feat = convert_categoric_one_hot_encoder(df_feat, 'CLASSE_PROCESSUAL')
    df_feat = convert_categoric_one_hot_encoder(df_feat, 'ASSUNTO_PROCESSUAL')
    df_feat = convert_categoric_one_hot_encoder(df_feat, 'CLASSIFICACAO_DA_UNIDADE')

    # Fill missing
    df_feat['PROCESSO_DIGITAL'] = df_feat['PROCESSO_DIGITAL'].fillna(1)
    df_feat = df_feat.fillna(df_feat.median())
        
    # Normalize columns
    df_feat = normalize_cols(df_feat, id_col).round(5)
    

    return df_feat