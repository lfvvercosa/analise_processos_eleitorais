import zipfile
import core.my_loader as my_loader
from core import my_log_orchestrator

# The event-log contains the procedural movements of each lawsuit ordered by its timestamp
# It will be used to the creation of n-gram later.

if __name__ == "__main__":
    base_path = 'dataset/'
    zip_file = 'tribunais_eleitorais/processos-tre-ne.zip'
    my_justice = 'TRIBUNAIS_ELEITORAIS'
    output_path = 'dataset/tribunais_eleitorais/tre-ne.xes'

    # Unzip dataset 
    # with zipfile.ZipFile(base_path + zip_file, 'r') as zip_ref:
    #     zip_ref.extractall(path=base_path)

    df_code_subj = my_loader.load_df_subject(base_path)
    df_code_type = my_loader.load_df_classes(base_path)
    df_code_mov = my_loader.load_df_movements(base_path)

    start_limit = 2017
    time_outlier = 0.03

    df, df_subj, df_mov = \
        my_log_orchestrator.load_dataframes(my_justice,
                                            ['TRE-NE'],
                                            base_path)

    df_mov = my_log_orchestrator.pre_process(
                df_subj,
                df_mov,
                df_code_subj,
                df_code_type,
                df_code_mov,
                start_limit,
                time_outlier
            )

    my_log_orchestrator.create_xes_file(df_mov, output_path)

    print('done!')