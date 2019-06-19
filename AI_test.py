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

data = pd.read_csv('housing.csv')
# print(data.shape)
# print(data.isnull().any().sum())

pd.plotting.scatter_matrix(data, alpha=0.7, figsize=(10,10), diagonal='kde')
plt.show()