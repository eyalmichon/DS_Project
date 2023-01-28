import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *
df_raw = pd.read_csv('rawdata.csv')
df_imputed = pd.read_csv('datatest.csv')

# accuracy = check_accuracy(df_raw, df_imputed, True)
# accuracy_no_nan = check_accuracy(df_raw, df_imputed, False)

# for percent in [0.1, 0.3, 0.5]:
#   for data_name in ['adultsIncome', 'housePrices', 'loans', 'pulsar']:
ML_accuracy_with_nan = [round(check_accuracy(df_raw, df_imputed ,0.1, True), 1),0,0,0]
ML_accuracy_no_nan = [round(check_accuracy(df_raw, df_imputed ,0.1, False), 1),0,0,0]
AR_accuracy_with_nan = [round(check_accuracy(df_raw, df_imputed ,0.1, True), 1),0,0,0]
AR_accuracy_no_nan = [round(check_accuracy(df_raw, df_imputed ,0.1, False), 1),0,0,0]

labels = ['Adults Income', 'House Prices', 'Loans', 'Pulsar']
x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, ML_accuracy_with_nan, width, label='Machine Learning with NaN')
rects2 = ax.bar(x - width/2, AR_accuracy_with_nan, width, label='Association Rules with NaN')
rects3 = ax.bar(x + width/2, ML_accuracy_no_nan, width, label='Machine Learning without NaN')
rects4 = ax.bar(x + 1.5 * width, AR_accuracy_no_nan, width, label='Association Rules without NaN')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of different methods')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)

fig.tight_layout()

plt.show()