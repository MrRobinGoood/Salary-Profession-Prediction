import pandas as pd
from clean_functions import clean_dataset
from tqdm.auto import tqdm

tqdm.pandas()

from multiprocessing import Pool

if __name__ == "__main__":
    data_names = ['TEST_RES.csv','TEST_SAL.csv']#'TRAIN_RES_2.csv', 'TRAIN_RES_3.csv', 'TRAIN_RES_4.csv',
                  #'TRAIN_RES_5.csv']  # , 'TRAIN_RES_1.csv','TRAIN_SAL.csv']
    for data_name in data_names:
        df = pd.read_csv(f'vprod_train/{data_name}', encoding='utf-8')

        names_for_clean = ['achievements_modified', 'demands']
        save_package_name = 'clean_res'
        if data_name == 'TEST_SAL.csv':
            save_package_name = 'clean_sal'
            names_for_clean = ['additional_requirements', 'education_speciality', 'other_vacancy_benefit',
                               'position_requirements',
                               'position_responsibilities', 'required_certificates']
        for_clean = df[names_for_clean]
        # разбиваем датасет на части
        processes = 12

        total_rows = len(for_clean)
        chunk_size = total_rows // processes
        print(chunk_size)

        pool = Pool(processes=processes)

        results = []
        for i in range(processes):
            if i == processes - 1:
                chunk = (chunk_size * i, total_rows)
            else:
                chunk = (chunk_size * i, chunk_size * (i + 1))
            print(chunk)
            results.append(pool.apply_async(clean_dataset, (for_clean, chunk)))

        pool.close()
        pool.join()

        print("Done!")

        final_cleaned_df = pd.DataFrame()
        for result in results:
            final_cleaned_df = pd.concat([final_cleaned_df, result.get()])

        df[names_for_clean] = final_cleaned_df
        df.to_csv(f'{save_package_name}/{data_name}', index=False, encoding='utf-8')
