import sys
import utils

sampled = True
sys.stdout = open('ml_result_sampled.txt', 'w')
for data_name in utils.Data_Names:
    utils.predict_nan_with_ml(data_name, sampled)
    for percent in utils.Percentages:
        df_no_nan = utils.pd.read_csv(f'./data/{data_name}/raw/{data_name}{"_sampled" if sampled else ""}_no_nan.csv')
        df_nan = utils.pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
        df_with_missing_ml = utils.pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ml_{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
        print('********************************************************')
        print(f'Accuracy for {percent} nan with ml = {utils.check_accuracy(df_no_nan, df_nan ,df_with_missing_ml,True)}')
        print(f'Accuracy for {percent} nan without Nan with ml = {utils.check_accuracy(df_no_nan, df_nan ,df_with_missing_ml,False)}')
sys.stdout.close()