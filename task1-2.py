import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
import seaborn as sns
import matplotlib.pyplot as plt

def confusion_metrics (conf_matrix):# save confusion matrix and slice into four pieces    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]    
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    print('---')
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))    # calculate f_1 score
    print(f'Accuracy: {round(conf_accuracy,4)}') 
    print(f'Sensitivity: {round(conf_sensitivity,4)}') 
    print(f'Specificity: {round(conf_specificity,4)}') 
    print(f'Precision: {round(conf_precision,4)}')

df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']
train = pd.read_csv('train.tsv', sep='\t', names=df_names)
train = train.dropna()
test = pd.read_csv('test.tsv', sep='\t', names=['Date', 'Temperature',
            'Humidity', 'Light', 'CO2', 'HumidityRatio'])
test = test.dropna()
results = pd.read_csv('results.tsv', sep='\t', names=['Occupancy'])
results = results.dropna()


#check what correlates with occupancy
corr = train.corr()
corr = corr.loc[:, 'Occupancy'].sort_values(ascending=False)
print(corr)

#first model - choosen variable, different than CO2 (light) 
X_train = train[['Light']]
y_train = train.Occupancy
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
cm = confusion_matrix(y_train, y_train_pred)
print('Light model - Training set metrics: ')
confusion_metrics(cm)
print(f"F score: {f1_score(y_train, y_train_pred)}")

X_test = test[['Light']]
y_test_pred = clf.predict(X_test)
cm = confusion_matrix(results, y_test_pred)
print('Light model - Test set metrics: ')
confusion_metrics(cm)
print(f"F score: {f1_score(results, y_test_pred)}")
#save results to csv
np.savetxt('out.tsv', y_test_pred, delimiter='\t')

# logistic regression classifier on all but date independent variables
clf_all = LogisticRegression()
X_train_all = train[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
clf_all.fit(X_train_all, y_train)
y_train_pred_all = clf_all.predict(X_train_all)
print('All variables model metrics: ')
cm = confusion_matrix(y_train, y_train_pred_all)
confusion_metrics(cm)
print(f"F score: {f1_score(y_train, y_train_pred_all)}")

# a) w pomieszczeniu zawsze powinien znajdować się człowiek, np. jest to cela więzienna (czujniki są wyłączane na czas spacerowania więźnia)
# false negative is less of a concern => precision more important => beta should be < 1
print(f"F score (b=0.3) light training set: {round(fbeta_score(y_train, y_train_pred, beta=0.3),3)}")
print(f"F score (b=0.3) light test set: {round(fbeta_score(results, y_test_pred, beta=0.3),3)}")
print(f"F score (b=0.3) all variables: {round(fbeta_score(y_train, y_train_pred_all, beta=0.3),3)}")

# b) w pomiesczeniu nie powinien znajdować się człowiek (urządzenie pełni funkcję alarmu)
# false negative is a concern => recall more important => beta should be > 1
print(f"F score (b=1.8) light training set: {round(fbeta_score(y_train, y_train_pred, beta=1.8),3)}")
print(f"F score (b=1.8) light test set: {round(fbeta_score(results, y_test_pred, beta=1.8),3)}")
print(f"F score (b=1.8) all variables: {round(fbeta_score(y_train, y_train_pred_all, beta=1.8),3)}")