from efficient_apriori import apriori
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datawig

Data_Names = ['adultsIncome', 'loans', 'heart', 'stroke']
Percentages = [0.1, 0.3, 0.5]

Params_for_module = {}
Params_for_module[f'adultsIncome{0.1}'] = 0.3, 0.6
Params_for_module[f'adultsIncome{0.3}'] = 0.25, 0.5
Params_for_module[f'adultsIncome{0.5}'] = 0.2, 0.4
Params_for_module[f'loans{0.1}'] = 0.2, 0.5
Params_for_module[f'loans{0.3}'] = 0.15, 0.45
Params_for_module[f'loans{0.5}'] = 0.1, 0.4
Params_for_module[f'heart{0.1}'] = 0.3, 0.5
Params_for_module[f'heart{0.3}'] = 0.25, 0.45
Params_for_module[f'heart{0.5}'] = 0.2, 0.4
Params_for_module[f'stroke{0.1}'] = 0.3, 0.5
Params_for_module[f'stroke{0.3}'] = 0.25, 0.45
Params_for_module[f'stroke{0.5}'] = 0.2, 0.4


def create_no_nan_data(data_name):
    path = f'./data/{data_name}/raw/raw_{data_name}.csv'
    df = pd.read_csv(path)

    df.replace(r"^ +| +$", r"", regex=True, inplace=True)
    df.replace('', np.nan, inplace=True)
    df.replace('?', np.nan, inplace=True)
    # drop columns with more than 50% missing values
    df = df.dropna(thresh=len(df)*0.5, axis=1)

    # # drop rows with more than 1 missing value
    df = df.dropna(thresh=len(df.columns), axis=0)

    df.to_csv(path.replace(f'raw_{data_name}',f'{data_name}_no_nan'), index=False)


def create_sampled_data(data_name):
    path = f'./data/{data_name}/raw/{data_name}_no_nan.csv'
    df = pd.read_csv(path)
    if len(df.index) > 1500:
        df = df.sample(n=1500, random_state=1)
    df.to_csv(path.replace('no_nan', 'sampled_no_nan'), index=False)


def create_nan_data(data_name):
    path = f'./data/{data_name}/raw/{data_name}_no_nan.csv'
    path_sampled = f'./data/{data_name}/raw/{data_name}_sampled_no_nan.csv'
    for percent in Percentages:
        df = pd.read_csv(path)
        df_sampled = pd.read_csv(path_sampled)
        
        df_with_missing = df.mask(np.random.rand(*df.shape) > 1-percent)
        df_with_missing_sampled = df_sampled.mask(np.random.rand(*df_sampled.shape) > 1-percent)

        # save nan data with missing values
        df_with_missing.to_csv(
            path.replace('raw', str(int(percent*100))+'percent', 1).replace('no_nan.csv', '', 1) + str(percent) + '_nan.csv', index=False)
        # save sampled data
        df_with_missing_sampled.to_csv(
            path_sampled.replace('raw', str(int(percent*100))+'percent', 1).replace('_sampled_no_nan.csv', '_sampled_', 1) + str(percent) + '_nan.csv', index=False)


def create_mapped_data(data_name):
    path = f'./data/{data_name}/raw/{data_name}_no_nan.csv'
    path_sampled = f'./data/{data_name}/raw/{data_name}_sampled_no_nan.csv'

    df = pd.read_csv(path)
    cols = create_map_from_data(df)
    # replace in raw data
    for col_name in cols:
        df.replace({col_name: cols[col_name]},inplace=True)
    df.to_csv(
        path.replace('_no_nan.csv','_mapped_no_nan.csv'), index=False)
    # replace in sampled raw data
    for col_name in cols:
        df.replace({col_name: cols[col_name]},inplace=True)
    df.to_csv(
        path_sampled.replace('_no_nan.csv','_mapped_no_nan.csv'), index=False)
    
    # replace in all percent data
    for percent in Percentages:
        path = f'./data/{data_name}/{int(percent * 100)}percent/{data_name}_{percent}_nan.csv'
        path_sampled = f'./data/{data_name}/{int(percent * 100)}percent/{data_name}_sampled_{percent}_nan.csv'
        df = pd.read_csv(path)
        # replace in nan data
        for col_name in cols:
            df.replace({col_name: cols[col_name]},inplace=True)
        df.to_csv(
            path.replace(f'_{percent}_nan.csv',f'_mapped_{percent}_nan.csv'), index=False)
        # replace in nan sampled data
        df = pd.read_csv(path_sampled)
        for col_name in cols:
            df.replace({col_name: cols[col_name]},inplace=True)
        df.to_csv(
            path_sampled.replace(f'_{percent}_nan.csv',f'_mapped_{percent}_nan.csv'), index=False)


# check the percentage of correct predictions, df1 is the complete data, df2 is the data with missing values that was imputed
def check_accuracy(df_no_nan, df_with_nan, df_predicted, with_nan=True):
    correct, total = 0, 0

    df_only_nan = df_with_nan.isnull()

    for i in tqdm(range(len(df_no_nan)), desc='Checking accuracy'):
        for j in range(len(df_no_nan.columns)):
            if df_only_nan.iloc[i,j]:
                if not with_nan and df_predicted.iloc[i, j] != df_predicted.iloc[i, j]:
                    continue
                elif df_no_nan.iloc[i, j] == df_predicted.iloc[i, j]:
                    correct += 1
                total += 1

    accuracy = 0 if total == 0 else correct/total
    # return the percentage of correct predictions
    return {'correct': correct, 'total': total, 'accuracy': np.round(np.abs(accuracy * 100), 5)}


def predict_nan_with_ml(data_name, sampled = False):

    df_original = pd.read_csv(f'./data/{data_name}/raw/{data_name}{"_sampled" if sampled else ""}_no_nan.csv')
    for percent in Percentages:
        print(f'******************\nImputing {data_name} with {percent} nan\n******************')
        path = f'./data/{data_name}/{int(percent * 100)}percent/{data_name}{"_sampled" if sampled else ""}_mapped_{percent}_nan.csv'
        df_with_missing = pd.read_csv(path)
        
        df_with_missing_imputed = datawig.SimpleImputer.complete(df_with_missing)
        
        # check if the df_original is type int, if so, convert the df_with_missing_imputed to int
        for col in df_original.columns:
            if df_original[col].dtype == 'int64' or df_original[col].dtype == 'object':
                df_with_missing_imputed[col] = round(df_with_missing_imputed[col]).astype(int)
        cols_map = create_map_from_data(df_original, reverse=True)
        for col in cols_map:
            df_with_missing_imputed.replace({col:cols_map[col]}, inplace=True)

        df_with_missing_imputed.to_csv(path.replace(f'{data_name}{"_sampled" if sampled else ""}_mapped', f'imputed_ml_{data_name}{"_sampled" if sampled else ""}'), index=False)
        


def predict_nan_with_ar(data_name, sampled = False):

    for percent in Percentages:
        print(f'******************\nImputing {data_name} with {percent} nan\n******************')
        path = f'./data/{data_name}/{int(percent * 100)}percent/{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv'
        df_with_missing = pd.read_csv(path)

        num_of_missing_before = df_with_missing.isnull().sum().sum()
        print(f'Number of missing values: {num_of_missing_before}')

        df_for_apriori = df_with_missing.copy(deep=True)

        # go through each column and replace the values with a string
        for col in df_for_apriori.columns:
            df_for_apriori[col] = df_for_apriori[col].astype(str)
        # convert the dataframe to a list of lists
        dict_for_apriori = df_for_apriori.to_dict(orient='records')
        transactions = [list(item.items()) for item in dict_for_apriori]
        min_support, min_confidence = Params_for_module[f'{data_name}{percent}'][0], Params_for_module[f'{data_name}{percent}'][1]
        print(f'min_support:{min_support}, min_confidence:{min_confidence}')
        itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, output_transaction_ids=False)
        rules = [rule for rule in rules if not any(map(lambda x: x[1] == 'nan', rule.rhs)) and not any(map(lambda x: x[1] == 'nan', rule.lhs))]
        print(f'We have {len(rules)} rules and here are the first 10:')
        sorted_rules = sorted(rules, key=lambda x: x.lift, reverse=True)
        for rule in sorted_rules[:10]:
            print(f'Rule: {rule.lhs} -> {rule.rhs}, Lift: {rule.lift}')
        df_missing_index_rows = df_with_missing.index[df_with_missing.isna().any(axis=1)]
            
        # create a dictionary with the column names and the index of the column
        col_names_dict = {}
        for i, col in enumerate(df_with_missing.columns):
            col_names_dict[col] = i

        # algortihm to fill missing values
        for index in tqdm(df_missing_index_rows):
            # print the current row on the same line (replacing the previous line)
            row = df_with_missing.iloc[index]
            rhs, lhs = [], []

            for col in row.index:
                rhs.append(col) if row[col] != row[col] else lhs.append(col)

            relevant_rules = []
            for rule in sorted_rules:
                # check if [col[0] for col in rule.rhs] is a subset of rhs
                if set([col[0] for col in rule.rhs]).issubset(set(rhs)): relevant_rules.append(rule)

            for rule in relevant_rules:
                # check if [keyval[0] for keyval in rule.lhs] is a subset of lhs
                if set([keyval for keyval in rule.lhs]).issubset(set([(col, row[col]) for col in lhs])):
                    should_fill = True
                    for keyval in rule.rhs:
                        if row[keyval[0]] == row[keyval[0]] and keyval[1] != row[keyval[0]]:
                            should_fill = False
                            break
                    if should_fill:
                        for keyval in rule.rhs:
                            df_with_missing.iloc[index, col_names_dict[keyval[0]]] = pd.to_numeric(keyval[1], errors='ignore')
        
        num_of_missing_after = df_with_missing.isnull().sum().sum()
        print(f'Number of filled values: {num_of_missing_before - num_of_missing_after}')
        print(f'Number of left missing values: {num_of_missing_after}')

        df_with_missing.to_csv(path.replace(f'{data_name}_', f'imputed_ar_{data_name}_'), index=False)

def print_graph_imputed_data(data_name, sampled = False):

    # labels = ['Adults Income', 'Loans', 'Heart', 'Stroke']
    labels = Percentages
    location = np.arange(len(labels))  # the label locations
    width = 0.20

    plt.rcParams["figure.figsize"] = (20, 10) 

    ML_accuracy_with_nan, ML_accuracy_no_nan = [], []
    AR_accuracy_with_nan, AR_accuracy_no_nan = [], []

    df_raw = pd.read_csv(f'./data/{data_name}/raw/{data_name}{"_sampled" if sampled else ""}_no_nan.csv')

    for percent in Percentages:
        df_nan = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
        df_ar = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ar_{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')
        df_ml = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ml_{data_name}{"_sampled" if sampled else ""}_{percent}_nan.csv')

        ML_accuracy_with_nan.append(check_accuracy(df_raw ,df_nan, df_ml, True)['accuracy'])
        ML_accuracy_no_nan.append(check_accuracy(df_raw ,df_nan, df_ml, False)['accuracy'])
        AR_accuracy_with_nan.append(check_accuracy(df_raw, df_nan, df_ar, True)['accuracy'])
        AR_accuracy_no_nan.append(check_accuracy(df_raw, df_nan, df_ar, False)['accuracy'])

    fig, ax = plt.subplots()

    rects1 = ax.bar(location - 1.5 * width, ML_accuracy_with_nan, width, label='Machine Learning with NaN')
    rects2 = ax.bar(location - width/2, AR_accuracy_with_nan, width, label='Association Rules with NaN')
    rects3 = ax.bar(location + width/2, ML_accuracy_no_nan, width, label='Machine Learning without NaN')
    rects4 = ax.bar(location + 1.5 * width, AR_accuracy_no_nan, width, label='Association Rules without NaN')    
            
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of different methods')
    ax.set_xticks(location, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    fig.tight_layout()

    # fig.set_size_inches(12, 6)

    plt.show()
    

def create_map_from_data(df, reverse=False):
    # Create a dictionary for all categorical columns
    cols = {}
    for col_name in df.columns:
        if(pd.api.types.is_string_dtype(df[col_name])):
            cols[col_name] = {}
            unique = df[col_name].unique()
            for j, val in enumerate(unique):
                if reverse:
                    cols[col_name][j] = val
                else:
                    cols[col_name][val] = j
    return cols  


def print_accuracy_for_all_datasets():
    for percent in Percentages:
        df_no_nan = pd.read_csv(f'./data/loans/raw/loans_no_nan.csv')
        # mapped_data = pd.read_csv(f'./data/loans/raw/loans_no_nan_mapped.csv')
        df_nan = pd.read_csv(f'./data/loans/{int(percent * 100)}percent/loans_{percent}_nan.csv')
        df_with_missing_ar = pd.read_csv(f'./data/loans/{int(percent * 100)}percent/imputed_ar_loans_{percent}_nan.csv')
        df_with_missing_ml = pd.read_csv(f'./data/loans/{int(percent * 100)}percent/imputed_ml_loans_{percent}_nan.csv')
        print(f'Accuracy for {percent} nan with ar = {check_accuracy(df_no_nan, df_nan ,df_with_missing_ar,True)}')
        print(f'Accuracy for {percent} nan without Nan with ar = {check_accuracy(df_no_nan, df_nan ,df_with_missing_ar,False)}')
        print('********************************************************')
        print(f'Accuracy for {percent} nan with ml = {check_accuracy(df_no_nan, df_nan ,df_with_missing_ml,True)}%')
        print(f'Accuracy for {percent} nan without Nan with ml = {check_accuracy(df_no_nan, df_nan ,df_with_missing_ml,False)}')
        print('********************************************************')