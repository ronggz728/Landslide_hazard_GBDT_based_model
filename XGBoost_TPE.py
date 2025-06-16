import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import xgboost as xgb
from sklearn import metrics
from pandas import DataFrame
import matplotlib.pyplot as plt
from hyperopt import fmin,tpe,hp,partial
# filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，
# header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
from sklearn.model_selection import train_test_split

test = pd.read_csv('F://fifth paper//输出统计//screened//XGBOOST//test.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI',  'plan',
                         'geological', 'rain', 'SPI', 'TWI', 'TRI', 'class'])

train = pd.read_csv('F://fifth paper//输出统计//screened//XGBOOST//train.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI',  'plan',
                         'geological', 'rain', 'SPI', 'TWI', 'TRI',  'class'])
target = 'class'
data = 'fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'plan',  'geological', 'rain', 'SPI', 'TWI', 'TRI'

x_columns_train = [x for x in train.columns if x not in [target]]
x_train = train[x_columns_train]
y_train = train['class']

x_columns_test = [x for x in test.columns if x not in [target]]
x_test = test[x_columns_test]
y_test = test['class']

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

clf = xgb.XGBClassifier()
clf.fit(x_train, y_train)

def percept(args):
    global x_train,y_train,y_test
    ppn = xgb.XGBClassifier(n_estimators = int(args["n_estimators"]),
                            max_depth=int(args['max_depth']),
                            learning_rate = args['learning_rate'],
                            colsample_bytree=args['colsample_bytree'],
                            subsample = args['subsample'],
                            random_state= 0)
    ppn.fit(x_train, y_train)
    y_pred = ppn.predict(x_test)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial,Trials
space = dict(n_estimators=hp.quniform("n_estimators", 50, 1000, 1),
             max_depth=hp.quniform('max_depth', 1,10,1),
             learning_rate=hp.uniform('learning_rate', 0.001, 1.0),
             colsample_bytree=hp.uniform('colsample_bytree', 0.5, 1.0),
             subsample=hp.uniform('subsample', 0.5, 0.8),
             random_state= 0)

tpe_algorithm = tpe.suggest
bayes_trials = Trials()
algo = partial(tpe.suggest,n_startup_jobs=100)
best = fmin(percept,space,algo = algo,max_evals=500,trials = bayes_trials)
print (best)
print (percept(best))
