import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

train_feature = np.load('./X_train_logistic.npy')
train_label = np.load('./y_train.npy')
test_feature = np.load('./X_test_logistic.npy')
test_label = np.load('./y_test.npy')

scaler = StandardScaler().fit(train_feature)
train_feature = scaler.transform(train_feature)
test_feature = scaler.transform(test_feature)

model = joblib.load('./logistic_parameter.pkl')

train_auroc = roc_auc_score(train_label, model.predict_proba(train_feature)[:, 1])
train_auprc = average_precision_score(train_label, model.predict_proba(train_feature)[:, 1])

test_auroc = roc_auc_score(test_label, model.predict_proba(test_feature)[:, 1])
test_auprc = average_precision_score(test_label, model.predict_proba(test_feature)[:, 1])

f = open("./20213334_logistic_regression.txt", "w")
f.write(f"20213334 \n {train_auroc} \n {train_auprc} \n {test_auroc} \n {test_auprc}")
f.close()