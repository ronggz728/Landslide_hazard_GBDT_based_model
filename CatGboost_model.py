import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from pandas import DataFrame
import evaluate_method as evaluate_method
import matplotlib.pyplot as plt
# filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，
# header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
from sklearn.model_selection import train_test_split

test = pd.read_csv('F://test.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])

train = pd.read_csv('F://train.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])
target = 'class'
data = 'fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan', 'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'

x_columns_train = [x for x in train.columns if x not in [target]]
x_train = train[x_columns_train]
y_train = train['class']

x_columns_test = [x for x in test.columns if x not in [target]]
x_test = test[x_columns_test]
y_test = test['class']

clf = CatBoostClassifier(logging_level='Verbose')
clf.fit(x_train, y_train)
predict_target = clf.predict(x_test)
expected = y_train
predicted = clf.predict(x_test)

probability_test = clf.predict_proba(x_test)
exp_test = DataFrame(probability_test)
exp_test.to_csv("F://OUTPUT//CB//CB_test.csv")
print(probability_test)

probability_train = clf.predict_proba(x_train)
exp_train = DataFrame(probability_train)
exp_train.to_csv("F://OUTPUT//CB//CB_train.csv")
print(probability_train)

print(metrics.classification_report(y_test, predict_target))
print(metrics.confusion_matrix(y_test, predict_target))

acc = evaluate_method.get_acc(y_test, predict_target)  # AUC value
test_auc = metrics.roc_auc_score(y_test, predict_target)
kappa = evaluate_method.get_kappa(y_test, predict_target)
IOA = evaluate_method.get_IOA(y_test, predict_target)
MCC = evaluate_method.get_mcc(y_test, predict_target)
recall = evaluate_method.get_recall(y_test, predict_target)
precision = evaluate_method.get_precision(y_test, predict_target)
f1 = evaluate_method.get_f1(y_test, predict_target)
MAPE = evaluate_method.get_MAPE(y_test, predict_target)

evaluate_method.get_ROC(y_test, predict_target,save_path='roc_stacking.txt')

print("ACC = " + str(acc))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))

'''
dataset_1 = pd.read_csv('F://第三篇//样本处理//调序删表名版//input_1.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_1 = [x for x in dataset_1.columns]
x_1 = dataset_1[x_columns_1]

dataset_2 = pd.read_csv('F://第三篇//样本处理//调序删表名版//input_2.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_2 = [x for x in dataset_2.columns]
x_2 = dataset_2[x_columns_2]

dataset_3 = pd.read_csv('F://第三篇//样本处理//调序删表名版//input_3.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_3 = [x for x in dataset_3.columns]
x_3 = dataset_3[x_columns_3]

dataset_4 = pd.read_csv('F://第三篇//样本处理//调序删表名版//input_4.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_4 = [x for x in dataset_4.columns]
x_4 = dataset_4[x_columns_4]

dataset_5 = pd.read_csv('F://第三篇//样本处理//调序删表名版//input_5.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_5 = [x for x in dataset_5.columns]
x_5 = dataset_5[x_columns_5]

probability_1 = clf.predict_proba(x_1)
exp_1 = DataFrame(probability_1)
exp_1.to_csv("F://fifth paper//OUTPUT//CB//CB_ALL_1.csv")
probability_2 = clf.predict_proba(x_2)
exp_2 = DataFrame(probability_2)
exp_2.to_csv("F://fifth paper//OUTPUT//CB//CB_ALL_2.csv")
probability_3 = clf.predict_proba(x_3)
exp_3 = DataFrame(probability_3)
exp_3.to_csv("F://fifth paper//OUTPUT//CB//CB_ALL_3.csv")
probability_4 = clf.predict_proba(x_4)
exp_4 = DataFrame(probability_4)
exp_4.to_csv("F://fifth paper//OUTPUT//CB//CB_ALL_4.csv")
probability_5 = clf.predict_proba(x_5)
exp_5 = DataFrame(probability_5)
exp_5.to_csv("F://fifth paper//OUTPUT//CB//CB_ALL_5.csv")

'''