# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import mglearn

cancer = load_breast_cancer()

"生成每个特征的两个类别的直方图"
#fig,axes = plt.subplots(15,2,figsize=(10,20))
#
#malignant = cancer.data[cancer.target == 0]
#benign = cancer.data[cancer.target == 1]
#
#ax = axes.ravel()
#
#for i in range(30):
#    _,bins = np.histogram(cancer.data[:,i],bins=50)
#    ax[i].hist(malignant[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
#    ax[i].hist(benign[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
#    ax[i].set_title(cancer.feature_names[i])
#    ax[i].set_yticks(())
#
#ax[0].set_xlabel("Feature magnitude")
#ax[0].set_ylabel("Frequency")
#ax[0].legend(["malignant", "benign"], loc="best")
#fig.tight_layout()


scaler = StandardScaler()
scaler.fit(cancer.data)

X_scaled = scaler.transform(cancer.data)

"保留数据的前两个主成分"
pca = PCA(n_components=2)
"对数据拟合PCA模型"
pca.fit(X_scaled)

"将数据变换到前两个主成分的方向上"
X_pca = pca.transform(X_scaled)

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
plt.legend(cancer.target_names,loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")