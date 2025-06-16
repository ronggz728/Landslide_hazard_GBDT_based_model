import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from ngboost import NGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from pandas import DataFrame
import matplotlib.pyplot as plt
from hyperopt import fmin,tpe,hp,partial
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

# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

clf = NGBClassifier(verbose= False)
clf.fit(x_train, y_train)

def percept(args):
    global x_train,y_train,y_test
    base_learner = DecisionTreeRegressor(max_depth=4)
    ppn = NGBClassifier(n_estimators = int(args["n_estimators"]),
                        learning_rate = args['learning_rate'],
                        #subsample = args['subsample'],
                        minibatch_frac=args['minibatch_frac'],
                        col_sample=args['col_sample'],
                        natural_gradient=False,
                        Base=base_learner,
                        random_state= 0)
    ppn.fit(x_train, y_train)
    y_pred = ppn.predict(x_test)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial,Trials
#n_estimators：迭代次数，即模型训练时的 boosting rounds 数量。
# learning_rate：学习速率，控制每个基学习器的贡献权重更新幅度。
#max_depth：基学习器（DT）的最大深度。
# minibatch_frac：行采样比例，用于随机梯度下降中的样本抽样。
# col_sample：列采样比例，用于随机梯度下降中的特征抽样。
space = dict(n_estimators=hp.quniform("n_estimators", 50, 1000, 1),
             learning_rate=hp.uniform('learning_rate', 0.001, 0.5),
             max_depth=hp.quniform('max_depth', 1,10,1),
             minibatch_frac=hp.uniform('minibatch_frac', 0.5,1),
             col_sample=hp.uniform('col_sample', 0.5,1),
             )

tpe_algorithm = tpe.suggest
bayes_trials = Trials()
algo = partial(tpe.suggest,n_startup_jobs=100)
best = fmin(percept,space,algo = algo,max_evals=500,trials = bayes_trials)
print (best)
print (percept(best))
