import pandas as pd
import numpy as np


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
