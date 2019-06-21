# coding=UTF-8
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
data = load_boston().data
data = pd.read_csv('housing.csv')
# print(data.shape)
# print(data.isnull().any().sum())
# pd.plotting.scatter_matrix(data, alpha=0.7, figsize=(10,10), diagonal='kde')
# plt.show()
x = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = data[['MEDV']]
SelectKBest = SelectKBest(f_regression, k=3)
bestFeature = SelectKBest.fit_transform(x,y)
SelectKBest.get_support()
x.columns[SelectKBest.get_support()]
features = data[['RM', 'PTRATIO', 'LSTAT']]
# pd.plotting.scatter_matrix(features, alpha=0.7, figsize=(6,6), diagonal='hist')
# plt.show()
scaler = MinMaxScaler()
for feature in features.columns:
    features['Standard'+feature] = scaler.fit_transform(features[[feature]])
# pd.plotting.scatter_matrix(features[['Standard_RM', 'Standard_PTRATIO', 'Standard_LSTAT']], alpha=0.7, figsize=(6,6), diagonal='hist')
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(features[['StandardRM', 'StandardPTRATIO', 'StandardLSTAT']], y, test_size=0.3,random_state=33)

# 線性回歸模型
lr = linear_model.LinearRegression()
lr_predict = cross_val_predict(lr,x_train, y_train, cv=5)
lr_score = cross_val_score(lr, x_train, y_train, cv=5)
lr_meanscore = lr_score.mean()

# SVR模型
linear_svr = SVR(kernel = 'linear')
linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
linear_svr_meanscore = linear_svr_score.mean()

poly_svr = SVR(kernel = 'poly')
poly_svr_predict = cross_val_predict(poly_svr, x_train, y_train, cv=5)
poly_svr_score = cross_val_score(poly_svr, x_train, y_train, cv=5)
poly_svr_meanscore = poly_svr_score.mean()

rbf_svr = SVR(kernel = 'rbf')
rbf_svr_predict = cross_val_predict(rbf_svr, x_train, y_train, cv=5)
rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
rbf_svr_meanscore = rbf_svr_score.mean()

# KNN模型
# score=[]
# for n_neighbors in range(1,21):
#     knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
#     knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
#     knn_score = cross_val_score(knn, x_train, y_train, cv=5)
#     knn_meanscore = knn_score.mean()
#     score.append(knn_meanscore)
# plt.plot(score)
# plt.xlabel('n-neighbors')
# plt.ylabel('mean-score')
# plt.show()
n_neighbors=2
knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
knn_score = cross_val_score(knn, x_train, y_train, cv=5)
knn_meanscore = knn_score.mean()

# 決策樹模型
# score=[]
# for n in range(1,11):
#     dtr = DecisionTreeRegressor(max_depth = n)
#     dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
#     dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
#     dtr_meanscore = dtr_score.mean()
#     score.append(dtr_meanscore)
# plt.plot(np.linspace(1,10,10), score)
# plt.xlabel('max_depth')
# plt.ylabel('mean-score')
# plt.show()
n=4
dtr = DecisionTreeRegressor(max_depth = n)
dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
dtr_meanscore = dtr_score.mean()

# 評分
evaluating = {'lr': lr_score,           # 線性回歸模型的分數
        'linear_svr': linear_svr_score, # 使用linear核的svr模型的分數
        'poly_svr': poly_svr_score,     # 使用poly核的svr模型的分數
        'rbf_svr': rbf_svr_score,       # 使用rbf核的svr模型的分數
        'knn': knn_score,               # knn模型的分數
        'dtr': dtr_score}               # 決策樹模型的分數
evaluating = pd.DataFrame(evaluating)
# evaluating.plot.kde(alpha=0.6,figsize=(8,7)) # 視覺化
# evaluating.hist(color='k',alpha=0.6,figsize=(8,7))
# plt.show()

# SVR-Linear優化
# lSVR_score=[]
# for i in [1,10,1e2,1e3,1e4]:
#     linear_svr = SVR(kernel = 'linear', C=i)
#     linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
#     linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
#     linear_svr_meanscore = linear_svr_score.mean()
#     lSVR_score.append(linear_svr_meanscore)
# plt.plot(lSVR_score)
# plt.show()
linear_svr = SVR(kernel = 'linear', C=10)
linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
linear_svr_meanscore = linear_svr_score.mean()

# SVR-Poly
# for i in [1,10,1e2,1e3,1e4]:
#     polySVR_score=[]
#     for j in np.linspace(1,10,10):
#         poly_svr = SVR(kernel = 'poly', C=i, degree=j)
#         poly_svr_predict = cross_val_predict(poly_svr, x_train, y_train, cv=5)
#         poly_svr_score = cross_val_score(poly_svr, x_train, y_train, cv=5)
#         poly_svr_meanscore = poly_svr_score.mean()
#         polySVR_score.append(poly_svr_meanscore)
#     plt.plot(polySVR_score, label='C='+str(i))
# plt.legend(loc='upper right')
# plt.xlabel('degree')
# plt.ylabel('score')
# plt.show()
poly_svr = SVR(kernel = 'poly', C=1000, degree=2)
poly_svr_predict = cross_val_predict(poly_svr, x_train, y_train, cv=5)
poly_svr_score = cross_val_score(poly_svr, x_train, y_train, cv=5)
poly_svr_meanscore = poly_svr_score.mean()

# SVE-Rbf
# for i in [1,10,1e2,1e3,1e4]:
#     rbfSVR_score=[]
#     for j in np.linspace(0.1,1,10):
#         rbf_svr = SVR(kernel = 'rbf', C=i, gamma=j)
#         rbf_svr_predict = cross_val_predict(rbf_svr, x_train, y_train, cv=5)
#         rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
#         rbf_svr_meanscore = rbf_svr_score.mean()
#         rbfSVR_score.append(rbf_svr_meanscore)
#     plt.plot(np.linspace(0.1,1,10),rbfSVR_score,label='C='+str(i))
#     plt.legend()
# plt.xlabel('gamma')
# plt.ylabel('score')
# plt.show()
rbf_svr = SVR(kernel = 'rbf', C=100, gamma=0.5)
rbf_svr_predict = cross_val_predict(rbf_svr, x_train, y_train, cv=5)
rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
rbf_svr_meanscore = rbf_svr_score.mean()

# 二次歸類
optimizer = { 'lr':lr_score,           # 線性回歸模型的分數
        'linear_svr':linear_svr_score, # 使用linear核的svr模型的分數 
        'poly_svr':poly_svr_score,     # 使用poly核的svr模型的分數
        'rbf_svr':rbf_svr_score,       # 使用rbf核的svr模型的分數
        'knn':knn_score,               # knn模型的分數
        'dtr':dtr_score }              # 決策樹模型的分數
optimizer = pd.DataFrame(optimizer)
# optimizer.hist(color='k',alpha=0.6,figsize=(8,7))
# optimizer.plot.kde(alpha=0.6,figsize=(8,7))
# plt.xlabel('score')
# plt.ylabel('Density')
# plt.show()

# 最優模型確定
optimizer.mean().sort_values(ascending = False)
# print(optimizer.mean().sort_values(ascending = False))

# 模型預測
# RBF
rbf_svr.fit(x_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)
rbf_svr_y_predict_score=rbf_svr.score(x_test, y_test)
# KNN
knn.fit(x_train,y_train)
knn_y_predict = knn.predict(x_test)
knn_y_predict_score = knn.score(x_test, y_test)
# poly_svr
poly_svr.fit(x_train,y_train)
poly_svr_y_predict = poly_svr.predict(x_test)
poly_svr_y_predict_score = poly_svr.score(x_test, y_test)
# dtr
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)
dtr_y_predict_score = dtr.score(x_test, y_test)
# lr
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
lr_y_predict_score = lr.score(x_test, y_test)
# linear_svr
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)
linear_svr_y_predict_score = linear_svr.score(x_test, y_test)
predict_score = {
        'lr':lr_y_predict_score,
        'linear_svr':linear_svr_y_predict_score,
        'poly_svr':poly_svr_y_predict_score,
        'rbf_svr':rbf_svr_y_predict_score,
        'knn':knn_y_predict_score,
        'dtr':dtr_y_predict_score
        }
predict_score = pd.DataFrame(predict_score, index=['score']).transpose()
predict_score.sort_values(by='score',ascending = False)
# print(predict_score.sort_values(by='score',ascending = False))

# 整理預測資料
plt.scatter(np.linspace(0,151,152), y_test, label='predict data')
labelname=[
        'rbf_svr_y_predict',
        'knn_y_predict',
        'poly_svr_y_predict',
        'dtr_y_predict',
        'lr_y_predict',
        'linear_svr_y_predict']
y_predict={
        'rbf_svr_y_predict':rbf_svr_y_predict,
        'knn_y_predict':knn_y_predict[:,0],
        'poly_svr_y_predict':poly_svr_y_predict,
        'dtr_y_predict':dtr_y_predict,
        'lr_y_predict':lr_y_predict[:,0],
        'linear_svr_y_predict':linear_svr_y_predict}
y_predict=pd.DataFrame(y_predict)
for name in labelname:
    plt.plot(y_predict[name],label=name)
plt.xlabel('predict data index')
plt.ylabel('target')
plt.legend()
plt.show()