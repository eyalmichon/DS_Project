import sys
import utils

sampled = False
sys.stdout = open('ar_result.txt', 'w')
for data_name in utils.Data_Names:
    utils.predict_nan_with_ar(data_name, sampled)
    for percent in utils.Percentages:
        print(f'************************** {data_name}, {percent} ***************************')
        df_no_nan = utils.pd.read_csv(f'./data/{data_name}/raw/{data_name}{"_sampled" if sampled else ""}_no_nan.csv')
        df_nan = utils.pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
        df_with_missing_ar = utils.pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ar_{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
        print(f'Accuracy for {percent} nan with ar = {utils.check_accuracy(df_no_nan, df_nan ,df_with_missing_ar,True)}\n')
        print(f'Accuracy for {percent} nan without Nan with ar = {utils.check_accuracy(df_no_nan, df_nan ,df_with_missing_ar,False)}\n')
sys.stdout.close()

