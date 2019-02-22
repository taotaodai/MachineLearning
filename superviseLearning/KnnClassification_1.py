# -*- coding: utf-8 -*-

"用K近邻算法预测鸢尾花的种类(n_neighbors为1个时)"
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

iris_dataset = load_iris()

#print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))

"描述"
#print(iris_dataset['DESCR'])

"样本数据"
#print(iris_dataset['data'])

"样本种类（花的名称）"
#print(iris_dataset['target_names'])

#print(iris_dataset['target'])

#print(iris_dataset['feature_names'])

"数据打乱并分组"
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

"绘制散点图"
#iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset['feature_names'])
#grr = pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)

"k近邻算法"
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=1,p=2,weights='uniform')


#X_new = X_test[0:1]
"对测试数据进行预测"
predction = knn.predict(X_test)
print("预测结果：{}".format(predction))
print("实际结果：{}".format(y_test))
print("准确率：{}".format(knn.score(X_test,y_test)))