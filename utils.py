from efficient_apriori import apriori
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datawig

Data_Names = ['adultsIncome', 'loans', 'pulsar', 'stroke']
Percentages = [0.1, 0.3, 0.5]

def create_nan_data(data_name):
    path = f'./data/{data_name}/raw/raw_{data_name}_no_nan.csv'
    for percent in Percentages:
        df = pd.read_csv(path)
        df_with_missing = df.mask(np.random.rand(*df.shape) > 1-percent)

        df_with_missing.to_csv(
            path.replace('raw', str(int(percent*100))+'percent', 1).replace('raw_','').replace('no_nan.csv', '', 1) + str(percent) + '_nan.csv', index=False)


def create_no_nan_data(data_name):
    path = f'./data/{data_name}/raw/raw_{data_name}.csv'
    df = pd.read_csv(path)

    df.replace('', np.nan, inplace=True)
    df.replace('?', np.nan, inplace=True)

    # drop columns with more than 50% missing values
    df = df.dropna(thresh=len(df)*0.5, axis=1)

    # # drop rows with more than 1 missing value
    df = df.dropna(thresh=len(df.columns), axis=0)

    df.to_csv(path.replace(f'raw_{data_name}',f'{data_name}_no_nan'), index=False)


# check the percentage of correct predictions, df1 is the complete data, df2 is the data with missing values that was imputed
def check_accuracy(df1, df2, percent, with_nan=True):
    correct, total = 0, 0
    for i in tqdm(range(len(df1)), desc='Checking accuracy'):
        for j in range(len(df1.columns)):
            if not with_nan and df2.iloc[i, j] != df2.iloc[i, j]:
                continue
            elif df1.iloc[i, j] == df2.iloc[i, j]:
                correct += 1
            total += 1

    # return the percentage of correct predictions
    return np.round(np.abs((((correct / total - (1 - percent)) / percent) * 100)), 5)


def predict_nan_with_ml(data_name):
    for percent in Percentages:
        print(f'******************\nImputing {data_name} with {percent} nan\n******************')
        path = f'./data/{data_name}/{int(percent * 100)}percent/{data_name}_{percent}_nan.csv'
        df_with_missing = pd.read_csv(path)
        df_with_missing_imputed = datawig.SimpleImputer.complete(
            df_with_missing, precision_threshold=0.05, num_epochs=100)
        df_with_missing_imputed.to_csv(path.replace(f'{data_name}_', f'imputed_ml_{data_name}_'), index=False)


def predict_nan_with_ar(data_name):
    for percent in Percentages:
        print(f'******************\nImputing {data_name} with {percent} nan\n******************')
        path = f'./data/{data_name}/{int(percent * 100)}percent/{data_name}_{percent}_nan.csv'
        df_with_missing = pd.read_csv(path)
        print(f'before the run: {df_with_missing.isnull().sum().sum()}')
        df_for_apriori = df_with_missing.copy(deep=True)

        # go through each column and replace the values with a string
        for col in df_for_apriori.columns:
            df_for_apriori[col] = df_for_apriori[col].astype(str)
        # convert the dataframe to a list of lists
        dict_for_apriori = df_for_apriori.to_dict(orient='records')
        transactions = [list(item.items()) for item in dict_for_apriori]
        itemsets, rules = apriori(transactions, min_support=0.2, min_confidence=0.55, output_transaction_ids=False)
        print(f'Found {len(rules)} rules')
        sorted_rules = sorted(rules, key=lambda x: x.lift, reverse=True)
        df_missing_index_rows = df_with_missing.index[df_with_missing.isna().any(
            axis=1)]
            
        # create a dictionary with the column names and the index of the column
        col_names = df_with_missing.columns
        col_names_dict = {}
        for i, col in enumerate(col_names):
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
                            df_with_missing.iloc[index, col_names_dict[keyval[0]]] = keyval[1]
        print(f'after the run: {df_with_missing.isnull().sum().sum()}')
        df_with_missing.to_csv(path.replace(f'{data_name}_', f'imputed_ar_{data_name}_'), index=False)


def write_output():
    sys.stdout = open('ar_result.txt', 'w')
    for data_name in Data_Names:
        df = pd.read_csv(f'./data/{data_name}/raw/{data_name}_no_nan.csv')
        for percent in Percentages:
            df_with_missing_ar = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ar_{data_name}_{percent}_nan.csv')
            real_percent = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/{data_name}_{percent}_nan.csv').isnull().sum().sum() / (df.size)
            print(f'******************\nChecking accuracy {data_name} with {percent} nan for ar\n\n******************')
            print(f'With NaN = {check_accuracy(df,df_with_missing_ar,real_percent,True)}%')
            print(f'Without NaN = {check_accuracy(df,df_with_missing_ar,real_percent,False)}%')
    sys.stdout.close()
    sys.stdout = open('ml_result.txt', 'w')
    for data_name in Data_Names:
        df = pd.read_csv(f'./data/{data_name}/raw/{data_name}_no_nan.csv')
        for percent in Percentages:
            df_with_missing_ml = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ml_{data_name}_{percent}_nan.csv')
            real_percent = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/{data_name}_{percent}_nan.csv').isnull().sum().sum() / (df.size)
            print(f'******************\nChecking accuracy {data_name} with {percent} nan for ml\n\n******************')
            print(f'With NaN = {check_accuracy(df,df_with_missing_ml,real_percent,True)}%')
            print(f'Without NaN = {check_accuracy(df,df_with_missing_ml,real_percent,False)}%')
    sys.stdout.close()

def print_graphs_all_imputed_data():

    labels = ['Adults Income', 'Loans', 'Pulsar', 'Stroke']
    location = np.arange(len(labels))  # the label locations
    width = 0.20 

    for percent in Percentages:
        
        ML_accuracy_with_nan, ML_accuracy_no_nan = [], []
        AR_accuracy_with_nan, AR_accuracy_no_nan = [], []

        for data_name in Data_Names:
            
            real_percent = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/{data_name}_{percent}_nan.csv').isnull().sum().sum() / (df.size)
            df_raw = pd.read_csv(f'./data/{data_name}/raw/raw_{data_name}_no_nan.csv')
            df_ar = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ar_{data_name}_{percent}_nan.csv')
            df_ml = pd.read_csv(f'./data/{data_name}/{int(percent * 100)}percent/imputed_ml_{data_name}_{percent}_nan.csv')

            ML_accuracy_with_nan.append(round(check_accuracy(df_raw, df_ml ,real_percent, True), 1))
            ML_accuracy_no_nan.append(round(check_accuracy(df_raw, df_ml ,real_percent, False), 1))
            AR_accuracy_with_nan.append(round(check_accuracy(df_raw, df_ar ,real_percent, True), 1))
            AR_accuracy_no_nan.append(round(check_accuracy(df_raw, df_ar ,real_percent, False), 1))

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
    
        plt.show()

def predict_all_datasets():
    for dataset in Data_Names:
        predict_nan_with_ar(dataset)
        predict_nan_with_ml(dataset)
    write_output()
    # print_graphs_all_imputed_data()

predict_all_datasets()
    
