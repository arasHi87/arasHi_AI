# AI從入門到放棄

[TOC]

## 前言
> *原文網址 https://hackmd.io/ZBcIaF0FRmiibOQ39EYv8w?both*

這是一篇讓你充滿絕望的教學，教你如何從入門到放棄，本篇目的是要教各位如何運用網路上的資料進行機器學習，以及一些資料處理的教學還有做機器學習時的小tip，並且會比較多種模型看看哪種的效果最好，我們主要會運用到以下的幾樣套件
* scikit-learn
這是我們要用來做機器學習的主體，同時也是我們獲取要做機器學習的資料的來源
* numpy
主要是一些數學運算之類的，因為它的基底是c++，比python還要快多了，所以大部分我們都會使用它(即使你不想用也不行，因為10套機器學習的套件裡有11套都需要用到)
* pandas
這是一個做資料處理的利器，基本上有資料處理的地方都會有它，常見的用途有整理資料、對資料裡的數據做統計、資料格式轉換......，除了不能讓資料變蘿莉以外它都做的到
* matplotlib
通常做完數據統計或整理之後我們都會想看他的分布之類的，這時候就是matplotlib出場的時候了，基本上統計學的圖都能畫出來，也能畫一些比較奇怪的圖，像是熱力圖之類的
## 事前準備
為了一勞永逸我們先一次把我們所需要的套件都先引入
```python=
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
```
* `sklearn.feature_selection`主要用途是在樣本集裡面降維或進行特徵選擇，其中`SelectKBest`是留下 topK 高分的特徵(後面會介紹甚麼是特徵)，`f_regression`則是作為對於回歸的計分函數，它會返回單一個 **p-value** (用於檢驗特徵與變量之間的相關性)，如果以上你都聽不懂，沒關係，你只要知道他們是拿來幫你判別做完機器學習後的模型的好壞的
* `sklearn.preprocessing`主要是在做特徵標準化的，講白話一點就是預處理，其中`MinMaxScaler`主要用途是把數據放大縮小到一個你指定的範圍裡，通常我們都會把數據縮放到0 - 1之間
* `train_test_split`分離訓練集以及測試集
* `cross_val_predict`交叉驗證的預測
* `cross_val_score`交叉驗證的評分
* `linear_model`為導入相關的算法類，可視情況導入不同的算法
* `SVR`是支持向量回歸的英文縮寫(support vector regression)，至於用途，就跟他名字一樣囉d(d＇∀＇)
* `KNeighborsRegressor`，中譯為K回歸模型，是一個無參數模型，主要原理是借助K個最近訓練樣本的目標數值，對待測樣本的回歸值進行決策，白話文就是根據樣本的的相似度進行預測回歸
* `DecisionTreeRegressor`一個叫回歸決策樹的神奇模型，至於內容是啥，因為比較複雜就不在這裡涉及了
* `load_boston` 導入波士頓房價的資料
## 特徵工程與資料集處理
### 資料導入與特徵處理
我們這次要做的主題是波士頓房價預測，首先我們要做的是獲得資料，我們有兩種作法，一種是用sklearn讀入資料，另外一種是從外部導入(下方的housing.csv可以在我的github找到)，這邊我們採用外部導入的方式。
```python=+
# 從sklearn導入
data = load_boston().data
# 從外部導入
data = pd.read_csv('housing.csv')
```
> In:  `data.shape`
> Out: `(506, 14)`

> In:  `data.isnull().any().sum()`
> Out: `0`

print 後我們可以看到上方的輸出，第一個輸出的前面的506是指有506筆資料，而後面的14是指我們的資料有14個維度，換個說法就是有14個特徵，而特徵的意思就是指我們要訓練的對象有何特徵可以判斷，舉個例子，我們要做的是房價預測，而我們的對象是房子本身的狀況，例如，屋齡、坪數......等，而這些就是我們所需要的特徵，詳細的資料跟每個特徵的意思參照下圖，而第二個輸出則是檢查數據中有無空值。
|Name|    |
|:----:|:----:|
|CRIM|城鎮人均犯罪率|
|ZN|住宅用地所占比例|
|INDUS|非住宅用地所占比例|
|CHAS|虛擬變量，用於迴歸分析|
|NOX|環保指數|
|RM|每棟住宅的房間數|
|AGE|1940以前建成的自住單位的比例|
|DIS|距離波士頓的五個工業中心的加權距離|
|RAD|距離高速公路的便利指數|
|TAX|每一萬美元的不動產稅率|
|PRTATIO|城鎮中的教師學生比例|
|B|城鎮中的黑人比例|
|LSTAT|地區中有多少房東屬於低收入族群|
|MEDV|自住房屋的均價|

接下來，我們可以將各特徵的分布用 matplotlib 畫出來觀察
```python=+
pd.plotting.scatter_matrix(data, alpha=0.7, figsize=(10,10), diagonal='kde')
plt.show()
```
下圖為各資料分布圖
![](https://i.imgur.com/KYYiSMV.png)
### 特徵選擇
因為特徵所包括的維度比較大總共有13維(不包含輸出的部分)，為了抱持我們的模型能夠進行高效率的運算，所以我們得選擇出相關性比較高的特徵。在這邊我們沒辦法單純依靠方差來判斷，因為每樣特徵都有自己的涵義，所以我們要找出與目標相關性比較強的變量作為最終變量。
```python=+
x = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = data[['MEDV']]
# k=3，選出三個相關性較高的特徵
SelectKBest = SelectKBest(f_regression, k=3)
# 擬合、訓練數據並且轉換
bestFeature = SelectKBest.fit_transform(x,y)
# 檢查transform後數據得轉變
print(bestFeature.shape)
# 取得特徵索引遮罩
SelectKBest.get_support()
# 留下我們要的特徵
x.columns[SelectKBest.get_support()]
```
> In:`SelectKBest.get_support()`
> Out: `[False False False False False True False False False False True False True]`

> In: `x.columns[SelectKBest.get_support()]`
> Out:` Index(['RM', 'PTRATIO', 'LSTAT'], dtype='object')`

在這邊我們可以看見與波士頓房價相連性最高的三個特徵分別為每棟住宅的房間數、城鎮中的教師學生比例、地區中有多少房東屬於低收入族群，思考一下可以發現其實他還蠻合理的，房間多的住宅價位一定比較高，越接近學區教師學生的比例就越高，而學區眾所皆知房價都比較貴，而當房東是低收入戶族群時，房價也會相對高一點，接下來我們在把我們選擇的特徵分布圖畫出來看看。
![](https://i.imgur.com/jp59xXe.png)
### 特徵歸一化
在做下面的code的時候你可能會噴出一堆warning告訴你，你正在試著從copy上覆蓋變量，那個不要裡他就好了(題外話，我被那個警告梗了半小時( |#`Д´)ﾉ )
``` python=+
scaler = MinMaxScaler()
for feature in features.columns:
    features['Standard_'+feature] = scaler.fit_transform(features[[feature]])
# 查看特徵歸一化後的數據
pd.plotting.scatter_matrix(features[['Standard_RM', 'Standard_PTRATIO', 'Standard_LSTAT']], alpha=0.7, figsize=(6,6), diagonal='hist')
plt.show()
```
我們可以注意雖然圖都長得一樣，但他們的值域範圍已經不相同了，下方這張圖的值域被壓到了0-1之間
![](https://i.imgur.com/3oEqhgp.png)
### 拆分數據集
接下來我們要先把數據拆分為訓練集以及測試集，原因是一方面能夠有獨立於訓練集以外的數據來評估學習後的模型，另一方面則是可以避免學習算法的過度擬和。
```python=+
x_train, x_test, y_train, y_test = train_test_split(features[['Standard_RM', 'Standard_PTRATIO', 'Standard_LSTAT']], y, test_size=0.3, random_state=33)
```
## 模型選擇與優化
我們總共採用四種模型來嘗試預測，並採用交叉驗證來進行評估，四種模型分別為下列四種，這邊先統一解釋一下，下方的cv參數是代表分成幾組的意思。
### 線性回歸模型
```python=+
lr = linear_model.LinearRegression()
lr_predict = cross_val_predict(lr,x_train, y_train, cv=5)
lr_score = cross_val_score(lr, x_train, y_train, cv=5)
lr_meanscore = lr_score.mean()
```
### 支持向量回歸模型-SVR
這邊我們分別嘗試看看三種核，分別是 linear、poly、rbf，這三種核有興趣的話可以上wiki看解釋，而kernel又是啥呢?在這邊稍微引用別人的描述

> ***Kernel trick在機器學習的角色就是希望當不同類別的資料在原始空間中無法被線性分類器區隔開來時，經由非線性投影後的資料能在更高維度的空間中可以更區隔開 - https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-kernel-%E5%87%BD%E6%95%B8-47c94095171 。***

白話文來說就是因為我們所要面對的問題並不會都呈線性分布，所我們通過Kernel來將其經由非線性投影，而使得我們能更加方便的處理。
```python=+
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
```
### KNN(K Nearest Neighbor) 模型
這邊我們先解釋一下下面的 n_neighbors 是甚麼，其實 n_neighbors 裡面開頭的 n 與 KNN 裡的 K 是同樣的意思，都是第 k/n 個，所以白話文就叫做第 N 個鄰居，也就是我們要選擇幾 K 個的意思。而因為我們無法確定當 N 等於多少時會獲得最佳結果，故我們先嘗試從1到20試試看，並將圖畫出來。
```python=+
score=[]
for n_neighbors in range(1,21):
    knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
    knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
    knn_score = cross_val_score(knn, x_train, y_train, cv=5)
    knn_meanscore = knn_score.mean()
    score.append(knn_meanscore)
plt.plot(score)
plt.xlabel('n-neighbors')
plt.ylabel('mean-score')
plt.show()
```
x 軸是我們的 N 取多少， y 軸是我們的平均得分，從下圖能發現模型的預測能力會隨著 N 上升而增加，但超過紅色圈起來的部分後便開始逐漸下降了，故我們的 N 取2。
![](https://i.imgur.com/5W7Ejua.jpg)
根據上面的選擇我們可以寫成下面的code。
```python=+
n_neighbors=2
knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
knn_score = cross_val_score(knn, x_train, y_train, cv=5)
knn_meanscore = knn_score.mean()
```
### 決策樹(Decision Tree)模型
跟上面的 KNN 模型一樣我們無法知道我們的決策樹的深度，故我們在遍歷一次觀察。
```python=+
score=[]
for n in range(1,11):
    dtr = DecisionTreeRegressor(max_depth = n)
    dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
    dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
    dtr_meanscore = dtr_score.mean()
    score.append(dtr_meanscore)
plt.plot(np.linspace(1,10,10), score)
plt.xlabel('max_depth')
plt.ylabel('mean-score')
plt.show()
```
x 軸是我們的最大深度， y 軸是我們的平均得分，我們能看到當最大深度為4時會有最好的預測能力。
![](https://i.imgur.com/4w9dxAV.jpg)
接著一樣在bang出一棵決策樹的模型(<ゝω・) 決策樹☆
```python=+
n=4
dtr = DecisionTreeRegressor(max_depth = n)
dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
dtr_meanscore = dtr_score.mean()
```
## 結果評估
接下來就是要分析各個模型做完的結果了━(ﾟ∀ﾟ)━( ﾟ∀)━( ﾟ)━( )━( )━(ﾟ)━(∀ﾟ)━(ﾟ∀ﾟ)━
### 最初結果
```python=+
evaluating = {'lr': lr_score,           # 線性回歸模型的分數
        'linear_svr': linear_svr_score, # 使用linear核的svr模型的分數
        'poly_svr': poly_svr_score,     # 使用poly核的svr模型的分數
        'rbf_svr': rbf_svr_score,       # 使用rbf核的svr模型的分數
        'knn': knn_score,               # knn模型的分數
        'dtr': dtr_score}               # 決策樹模型的分數
evaluating = pd.DataFrame(evaluating)
evaluating.plot.kde(alpha=0.6,figsize=(8,7)) # 視覺化
plt.show()
```
所有模型得分密度圖以及長條圖，基本上我們能看到我們的svr幾乎是爛的，下面再看看我們能不能救救他吧╮(╯_╰)╭
![](https://i.imgur.com/HCB3DKt.png)
```python=+
evaluating.hist(color='k',alpha=0.6,figsize=(8,7))
plt.show()
```
![](https://i.imgur.com/42QTYcb.png)
## SVR模型優化
### SVR-Linear優化
這裡我們要先提到 SVR 裡的一個重要參數，`懲罰參數C`，白話文來說就是對誤差的容忍度，當 C 越高時則說明越不能接受誤差出現容易過度擬合，當 C 過小時則對誤差的寬容越高容易欠擬合。C 過大或過小泛化能力都會變差，sklearn在線性核裡默認的 C 為1。所以對於線性核的優化我們能從懲罰參數下手。我們先看看C分別位於1、10、100、1000的影響，x 軸是log10( C )，y軸是分數。
```python=+
lSVR_score=[]
for i in [1,10,1e2,1e3,1e4]:
    linear_svr = SVR(kernel = 'linear', C=i)
    linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
    linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
    linear_svr_meanscore = linear_svr_score.mean()
    lSVR_score.append(linear_svr_meanscore)
plt.plot(lSVR_score)
plt.show()
```
![](https://i.imgur.com/fL4U9di.png)
我們能發現當C為10時便會趨於極值，故我們選擇10為其懲罰係數。
```python=+
linear_svr = SVR(kernel = 'linear', C=10)
linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
linear_svr_meanscore = linear_svr_score.mean()
```
### SVR-Poly優化
這邊我們除了更改懲罰係數外我們也試著修改深度看看
```python=+
polySVR_score=[]
for i in [1,10,1e2,1e3,1e4]:
    for j in np.linspace(1,10,10):
        poly_svr = SVR(kernel = 'poly', C=i, degree=j)
        poly_svr_predict = cross_val_predict(poly_svr, x_train, y_train, cv=5)
        poly_svr_score = cross_val_score(poly_svr, x_train, y_train, cv=5)
        poly_svr_meanscore = poly_svr_score.mean()
        polySVR_score.append(poly_svr_meanscore)
plt.plot(polySVR_score)
plt.show()
```
我們能發現當 C 大於10普遍分數會較高，degree為2時分數為最高(預設為C=1、degree=3)
![](https://i.imgur.com/44JiKZp.png)
優化後模型分數增加不少
```python=+
poly_svr = SVR(kernel = 'poly', C=1000, degree=2)
poly_svr_predict = cross_val_predict(poly_svr, x_train, y_train, cv=5)
poly_svr_score = cross_val_score(poly_svr, x_train, y_train, cv=5)
poly_svr_meanscore = poly_svr_score.mean()
```
### SVR-Rbf優化
這邊我們使用 C 以及gamma參數進行優化，gamma是選擇RBF函數作為kernel後，該函數自帶的一個參數。隱含地決定了數據映射到新的特徵空間後的分佈，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的個數影響訓練與預測的速度。
```python=+
for i in [1,10,1e2,1e3,1e4]:
    rbfSVR_score=[]
    for j in np.linspace(0.1,1,10):
        rbf_svr = SVR(kernel = 'rbf', C=i, gamma=j)
        rbf_svr_predict = cross_val_predict(rbf_svr, x_train, y_train, cv=5)
        rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
        rbf_svr_meanscore = rbf_svr_score.mean()
        rbfSVR_score.append(rbf_svr_meanscore)
    plt.plot(np.linspace(0.1,1,10),rbfSVR_score,label='C='+str(i))
    plt.legend()
plt.xlabel('gamma')
plt.ylabel('score')
plt.show()
```
下圖我們能觀察到當gamma漸增時，得分也會跟著增長，但當C>=10時影響就較小了，當C=100，gamma=0.5時能獲得最佳模型。
![](https://i.imgur.com/UQ4svf7.png)
優化的啦( ఠൠఠ )ﾉ
```python=+
rbf_svr = SVR(kernel = 'rbf', C=100, gamma=0.5)
rbf_svr_predict = cross_val_predict(rbf_svr, x_train, y_train, cv=5)
rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
rbf_svr_meanscore = rbf_svr_score.mean()
```
### 二次分類
不多說了( ﾟ Χ ﾟ)
```python=+
optimizer = { 'lr':lr_score,           # 線性回歸模型的分數
        'linear_svr':linear_svr_score, # 使用linear核的svr模型的分數 
        'poly_svr':poly_svr_score,     # 使用poly核的svr模型的分數
        'rbf_svr':rbf_svr_score,       # 使用rbf核的svr模型的分數
        'knn':knn_score,               # knn模型的分數
        'dtr':dtr_score }              # 決策樹模型的分數
optimizer= pd.DataFrame(optimizer)
optimizer.plot.kde(alpha=0.6,figsize=(8,7))
plt.xlabel('score')
plt.ylabel('Density')
plt.show()
```
一樣把它畫出來的啦!
![](https://i.imgur.com/wmUkyOh.png)
```python=+
optimizer.hist(color='k',alpha=0.6,figsize=(8,7))
plt.show()
```
![](https://i.imgur.com/Ct6rpG0.png)
## 最終結果
### 最優模型確定
對比每個經過優化的模型的分數。
```python=+
print(optimizer.mean().sort_values(ascending = False))
```
|Type|Score|
|:--:|:---:|
|rbf_svr|0.775660|
|knn|0.775493|
|poly_svr|0.772448|
|dtr|0.708923|
|lr|0.665095|
|linear_svr|0.663172|
|dtype|float64|
此時發現，rbf核的SVR模型在優化後變成了最好的模型。線性核的SVR和線性回歸因為策略的局限性預測能力排在最後。
### 模型預測
接下來我們要嘗試預測數據集。
```python=+
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
print(predict_score.sort_values(by='score',ascending = False))
```
預測結果的排名
|type|score|
|:--:|:---:|
|rbf_svr|0.737578|
|dtr|0.710227|
|poly_svr|0.708969|
|knn|0.695671|
|linear_svr|0.637349|
|lr|0.618017|
各個模型的預測值整理
![](https://i.imgur.com/nzec6OS.png)
## 結論
~~我不做人啦jojo~~，在經過各種奇妙的優化後，原本跟我校排一樣爛的RBF-SVR模型變成第一了，其實還有一些特徵優化能做，但這日後再談了Z(_ _)Z
