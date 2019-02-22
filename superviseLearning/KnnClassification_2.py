# -*- coding: utf-8 -*-

"用K近邻算法预测乳腺肿瘤是恶性还是良性(n_neighbors为1个到多个)"
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#X,y = mglearn.datasets.make_wave(n_samples=40)
#plt.plot(X,y,'o')
#plt.ylim(-3,3)
#plt.xlabel("Feture")
#plt.ylabel("Target")

cancer = load_breast_cancer()

#print(cancer.keys())

#print(cancer['data'].shape)

#print(cancer['target'])

#print(cancer['target_names'])

#print(cancer['feature_names'])

#mglearn.plots.plot_knn_classification(n_neighbors=1)


X_train,X_test,y_train,y_test = train_test_split(cancer['data'],cancer['target'],stratify=cancer['target'],random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_setting = range(1,11)

"训练集和测试集n_neighbors在1~10范围内的准确率对比"
for n_neighbors in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_setting,training_accuracy,label="training accuracy")
plt.plot(neighbors_setting,test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()





