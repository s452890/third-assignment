import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']
train = pd.read_csv('train.tsv', sep='\t', names=df_names)
test = pd.read_csv('test.tsv', sep='\t', names=df_names)
train = train.dropna()

#check what correlates with occupancy
corr = train.corr()
corr = corr.loc[:, 'Occupancy'].sort_values(ascending=False)
print(corr)

X_train = train[['Light']]
Y_train = train.Occupancy

