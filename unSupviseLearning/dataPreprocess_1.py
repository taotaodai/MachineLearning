# -*- coding: utf-8 -*-

"非监督学习-数据预处理"
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

cancer = load_breast_cancer()

X_train,X_test,y_train,y_test = train_test_split(cancer['data'],cancer['target'],random_state=1)

"创建缩放器，下面有四种数据变换方式"
scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = Normalizer()

scaler.fit(X_train)

"设置缩放器，使所有feature的值在0~1范围内"
MinMaxScaler(copy=True,feature_range=(0,1))

"对训练集进行变换数据(缩放)"
X_train_scaled = scaler.transform(X_train)

"对测试集进行变换数据(缩放)"
X_test_scaled = scaler.transform(X_test)

svm_before = SVC(C=100)
svm_before.fit(X_train,y_train)

print("数据未缩放前的准确率：{}".format(svm_before.score(X_test,y_test)))

svm_after = SVC(C=100)
svm_after.fit(X_train_scaled,y_train)
print("数据缩放后的准确率：{}".format(svm_after.score(X_test_scaled,y_test)))