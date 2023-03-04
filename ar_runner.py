import sys
import utils

sampled = False
print_accuracy = False
sys.stdout = open(f'ar_result{"_sampled" if sampled else ""}{"_acc" if print_accuracy else ""}.txt', 'w')
for data_name in ['stroke']:
    utils.predict_nan_with_ar(data_name, sampled)
    if print_accuracy:
        for percent in utils.Percentages:
            print(f'************************** {data_name}, {percent} ***************************')
            df_no_nan = utils.pd.read_csv(f'./data/{data_name}/raw/{data_name}{"_sampled" if sampled else ""}_no_nan.csv')
            df_nan = utils.pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
            df_with_missing_ar = utils.pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ar_{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
            print(f'Accuracy for {percent} nan without Nan with ar = {utils.check_accuracy(df_no_nan, df_nan ,df_with_missing_ar,False)}')
            print(f'Accuracy for {percent} nan with ar = {utils.check_accuracy(df_no_nan, df_nan ,df_with_missing_ar,True)}')
sys.stdout.close()

