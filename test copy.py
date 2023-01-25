import pandas as pd
import numpy as np

path = './data/pulsar/raw/raw_pulsar_no_nan.csv'
percent = 0.5
df = pd.read_csv(path)
max_remove = int(len(df.columns) * 0.5)


print(f'Original shape: {df.shape}')


def get_random():
    while True:
        random_row = np.random.randint(0, len(df))
        random_col = np.random.randint(0, len(df.columns))
        if df.at(random_row).isna().sum() < max_remove:
            return random_row, random_col


for i in range(df.size*percent):
    row, col = get_random()
    df.at[row, col] = np.nan


df.to_csv(
    path.replace('raw', str(percent*100)+'percent', 1).replace('.csv', '', 1) + '_' + str(percent) + 'nan.csv', index=False)

# df.replace('', np.nan, inplace=True)
# df.replace('?', np.nan, inplace=True)

# print(df.shape)
# # drop columns with more than 50% missing values
# df = df.dropna(thresh=len(df)*0.5, axis=1)

# # drop rows with more than 1 missing value
# df = df.dropna(thresh=len(df.columns), axis=0)
# print(df.shape)
