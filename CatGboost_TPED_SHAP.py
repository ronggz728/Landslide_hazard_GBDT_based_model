import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from pandas import DataFrame
import evaluate_method as evaluate_method
import matplotlib.pyplot as plt
import shap

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，
# header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
from sklearn.model_selection import train_test_split

test = pd.read_csv('F://test.csv', header=None, sep=',',
                 names=['Distance to faults', 'Distance to road', 'Distance to river', 'Lithology', 'Elevation', 'Slope', 'NDVI', 'Profile curve', 'Plan curve',
                        'Aspect', 'Geological age', 'AAP', 'SPI', 'TWI', 'TRI', 'STI', 'Land cover', 'Class'])

train = pd.read_csv('F://train.csv', header=None, sep=',',
                 names=['Distance to faults', 'Distance to road', 'Distance to river', 'Lithology', 'Elevation', 'Slope', 'NDVI', 'Profile curve', 'Plan curve',
                        'Aspect', 'Geological age', 'AAP', 'SPI', 'TWI', 'TRI', 'STI', 'Land cover', 'Class'])
target = 'Class'
data = 'Distance to faults', 'Distance to road', 'Distance to river', 'Lithology', 'Elevation', 'Slope', 'NDVI', 'Profile curve', 'Plan curve', 'Aspect', 'Geological age', 'AAP', 'SPI', 'TWI', 'TRI', 'STI', 'Land cover', 'Class'

x_columns_train = [x for x in train.columns if x not in [target]]
x_train = train[x_columns_train]
y_train = train['Class']

x_columns_test = [x for x in test.columns if x not in [target]]
x_test = test[x_columns_test]
y_test = test['Class']

clf = CatBoostClassifier(n_estimators=334, learning_rate=0.251699502, max_depth=10, bagging_temperature=8.652005041, bootstrap_type='Bayesian',  logging_level='Verbose')
clf.fit(x_train, y_train)
predict_target = clf.predict(x_test)
expected = y_train
predicted = clf.predict(x_test)

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

#SHAP部分

X = pd.read_csv('F://X_ALL.csv', header=None, sep=',',
                 names=['Distance to faults', 'Distance to road', 'Distance to river', 'Lithology', 'Elevation', 'Slope', 'NDVI', 'Profile curve', 'Plan curve',
                        'Aspect', 'Geological age', 'AAP', 'SPI', 'TWI', 'TRI', 'STI', 'Land cover'])

y = pd.read_csv('F://Y_ALL.csv', header=None, sep=',', names=['class'])
x_train = pd.DataFrame(x_train)
model = clf
explainer = shap.TreeExplainer(model)
# 计算训练数据的SHAP值
shap_values = explainer.shap_values(x_train)

# 对于二分类问题，shap_values 会返回两个数组，每个数组对应一个类别的SHAP值
# 我们取第二个类别（通常是正类）的SHAP值
if len(shap_values) > 1:
    shap_values_class = shap_values[1]
else:
    # 如果shap_values没有两个元素，那么它就是所有类别的汇总
    shap_values_class = shap_values[0]
# 检查 expected_value 的类型
if isinstance(explainer.expected_value, list):
    base_value = explainer.expected_value[1]  # 对于二分类问题，取正类的基线值
else:
    base_value = explainer.expected_value  # 如果是标量值，则直接使用

# 创建SHAP的Explanation对象
shap_values_obj = shap.Explanation(values=shap_values_class,
                                   base_values=base_value,
                                   data=x_train.values,
                                   feature_names=x_train.columns)

# 使用SHAP的Explanation对象生成force_plot
shap.force_plot(base_value, shap_values_class, x_train.iloc[0,:])

# 计算每个特征的SHAP值的平均绝对值，作为特征重要性
feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': np.abs(shap_values).mean(0)
}).sort_values('importance', ascending=False)

# 打印特征重要性
print(feature_importance)

print('解释模型的常数:', explainer.expected_value)
print('训练样本预测值的log odds ratio的均值:', np.log(model.predict_proba(x_train)[:, 1]/ (1 - model.predict_proba(x_train)[:, 1])).mean())
print('常数与归因值之和:', explainer.expected_value + shap_values[0].sum())
print('预测值:', model.predict_proba(x_train.iloc[0:1])[:, 1][0])
print('预测值的log odds ratio:', np.log(model.predict_proba(x_train.iloc[0:1])[:, 1][0] / (1 - model.predict_proba(x_train.iloc[0:1])[:, 1][0])))

# 初始化 SHAP 的 JavaScript 环境
shap.initjs()
# 生成 Force Plot，设置 show=False
shap.force_plot(explainer.expected_value, shap_values[0], x_train.iloc[0, :], matplotlib=True, show=False)

# 保存图片
plt.savefig("F://OUTPUT//SHAP//force_plot.jpg", dpi=1000, bbox_inches='tight')
plt.close()

sample_index = 400  # 选择一个样本索引进行解释
shap.force_plot(explainer.expected_value, shap_values[sample_index], x_train.iloc[sample_index],matplotlib=True, show=False)
# 保存图片
plt.savefig("F://OUTPUT//SHAP//400_index.jpg", dpi=1000, bbox_inches='tight')
plt.close()

sample_0_shap = pd.DataFrame(x_train.iloc[0,:])
sample_0_shap.rename(columns={0: 'feature_value'}, inplace=True)
sample_0_shap['shap_value'] = shap_values[0]
sample_0_shap.sort_values('shap_value', ascending=False)

shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
# 保存图片
plt.savefig("F://OUTPUT//SHAP//summary_bar.jpg", dpi=1000, bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values, x_train, show=False)
# 保存图片
plt.savefig("F://OUTPUT//SHAP//summary.jpg", dpi=1000, bbox_inches='tight')
plt.close()

feature_importance = pd.DataFrame()
feature_importance['feature'] = x_train.columns
feature_importance['importance'] = np.abs(shap_values).mean(0)
feature_importance.sort_values('importance', ascending=False)

shap.dependence_plot('Lithology', shap_values, x_train, interaction_index=None, show=False)
# 保存图片
plt.savefig("F://OUTPUT//SHAP//Lithology_dependance.jpg", dpi=1000, bbox_inches='tight')
plt.close()

shap.dependence_plot('Elevation', shap_values, x_train, interaction_index=None, show=False)
# 保存图片
plt.savefig("F://OUTPUT//SHAP//Elevation_dependance.jpg", dpi=1000, bbox_inches='tight')
plt.close()

plt.figure(figsize=(7.5, 5))
plt.scatter(x_train['Lithology'], shap_values[:, 0], s=10, alpha=1)

shap.dependence_plot('Lithology', shap_values, x_train, interaction_index='Elevation',show=False)
# 保存图片
plt.savefig("F://OUTPUT//SHAP//Lithology_Elevation.jpg", dpi=1000, bbox_inches='tight')
plt.close()


# 创建 shap.Explanation 对象
shap_explanation = shap.Explanation(values=shap_values[50:672,:],
                                    base_values=explainer.expected_value,
                                    data=x_train.iloc[50:672,:], feature_names=x_train.columns)
# 绘制热图
shap.plots.heatmap(shap_explanation,show=False)
# 保存图片
plt.savefig("F://OUTPUT//SHAP//heatmap.jpg", dpi=1000, bbox_inches='tight')
plt.close()
try:
    shap_interaction_values = explainer.shap_interaction_values(x_train)
    print("交互效应的SHAP值计算成功。")
except AttributeError:
    print("当前解释器不支持计算交互效应的SHAP值。")
#shap.dependence_plot(('elevation', 'lithology'), shap_interaction_values, x_train, interaction_index=None)
#plt.figure(figsize=(7.5, 5))
#plt.scatter(x_train['lithology'], shap_interaction_values[:, 0, 0], s=10, alpha=1)
