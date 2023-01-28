from efficient_apriori import apriori
import pandas as pd
import numpy as np
from tqdm import tqdm
# import datawig


def create_train_test_from_data():
    def splitOnPersent(path):

        df = pd.read_csv(path)

        df_train = df.sample(frac=0.75, random_state=0)
        df_test = df.drop(df_train.index)

        df_train.to_csv(path.replace('raw', 'train', 1), index=False)
        df_test.to_csv(path.replace('raw', 'test', 1), index=False)

    for data_name in ['adultsIncome', 'housePrices', 'loans', 'pulsar']:
        for percent in [0.1, 0.3, 0.5]:
            path = f'./data/{data_name}/{int(percent * 100)}percent/raw_{data_name}_{percent}nan.csv'
            splitOnPersent(path)


def create_nan_data():
    # ['adultsIncome', 'housePrices', 'loans', 'pulsar']
    data_name = 'adultsIncome'
    path = f'./data/{data_name}/raw/raw_{data_name}_no_nan.csv'
    # [0.1,0.3,0.5]:
    percent = 0.5
    df = pd.read_csv(path)
    max_remove = int(len(df.columns) * 0.5)

    print(f'Original shape: {df.shape}')

    counter = {}

    def get_random():
        while True:
            random_row = np.random.randint(0, len(df))
            random_col = np.random.randint(0, len(df.columns))
            if df.iloc[random_row].isna().sum() < max_remove:
                return random_row, random_col

            # count how many times we failed to find a row with less than max_remove nan values
            if counter.get(random_row) is None:
                counter[random_row] = 1
            else:
                counter[random_row] += 1
                print(
                    f'Tried {counter[random_row]} times to find a row with less than {max_remove} nan values')

    for i in range(int(df.size*percent)):

        print(f'Iteration {i+1} of {int(df.size*percent)}')
        row, col = get_random()
        df.iloc[row, col] = np.nan

    print(f'New shape: {df.shape}')

    df.to_csv(
        path.replace('raw', str(int(percent*100))+'percent', 1).replace('no_nan.csv', '', 1) + str(percent) + 'nan.csv', index=False)


def create_no_nan_data():
    # ['adultsIncome', 'housePrices', 'loans', 'pulsar']
    data_name = 'adultsIncome'
    path = f'./data/{data_name}/raw/raw_{data_name}.csv'
    df = pd.read_csv(path)

    df.replace('', np.nan, inplace=True)
    df.replace('?', np.nan, inplace=True)

    print(df.shape)
    # drop columns with more than 50% missing values
    df = df.dropna(thresh=len(df)*0.5, axis=1)

    # drop rows with more than 1 missing value
    df = df.dropna(thresh=len(df.columns), axis=0)
    print(df.shape)


# check the percentage of correct predictions, df1 is the complete data, df2 is the data with missing values that was imputed
def check_accuracy(df1, df2, percent, with_nan=True):
    correct = 0
    total = 0
    for i in range(len(df1)):
        for j in range(len(df1.columns)):
            if not with_nan and df2.iloc[i, j] != df2.iloc[i, j]:
                continue
            elif df1.iloc[i, j] == df2.iloc[i, j]:
                correct += 1
            total += 1

    # return the percentage of correct predictions
    return ((correct / total - (1 - percent)) / percent) * 100


def predict_nan_with_ml():
    for data_name in ['adultsIncome', 'housePrices', 'loans', 'pulsar']:
        for percent in [0.1, 0.3, 0.5]:
            print(
                f'******************\n\nImputing {data_name} with {percent} nan\n\n******************')
            path = f'./data/{data_name}/{int(percent * 100)}percent/raw_{data_name}_{percent}nan.csv'
            df_with_missing = pd.read_csv(path)
            df_with_missing_imputed = datawig.SimpleImputer.complete(
                df_with_missing, precision_threshold=0.05, num_epochs=100)
            df_with_missing_imputed.to_csv(
                path.replace('raw', 'imputed_ml', 1), index=False)


def predict_nan_with_ar():
    for data_name in ['adultsIncome', 'housePrices', 'loans', 'pulsar']:
        for percent in [0.1, 0.3, 0.5]:
            print(
                f'******************\n\nImputing {data_name} with {percent} nan\n\n******************')
            path = f'./data/{data_name}/{int(percent * 100)}percent/raw_{data_name}_{percent}nan.csv'
            df_with_missing = pd.read_csv(path)
            df_for_apriori = df_with_missing.copy()

            # go through each column and replace the values with a string
            for col in df_for_apriori.columns:
                df_for_apriori[col] = df_for_apriori[col].astype(str)
            # convert the dataframe to a list of lists
            dict_for_apriori = df_for_apriori.to_dict(orient='records')
            transactions = [list(item.items()) for item in dict_for_apriori]
            itemsets, rules = apriori(
                transactions, min_support=0.3, min_confidence=0.6, output_transaction_ids=False)
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
                rhs = []
                lhs = []
                for col in row.index:
                    if row[col] != row[col]:
                        rhs.append(col)
                    else:
                        lhs.append(col)
                relevant_rules = []
                for rule in sorted_rules:
                    # check if [col[0] for col in rule.rhs] is a subset of rhs
                    if set([col[0] for col in rule.rhs]).issubset(set(rhs)):
                        relevant_rules.append(rule)
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
                                df_with_missing.iloc[index,
                                                     col_names_dict[keyval[0]]] = keyval[1]

            df_with_missing.to_csv(
                path.replace('raw', 'imputed_ar', 1), index=False)


predict_nan_with_ar()