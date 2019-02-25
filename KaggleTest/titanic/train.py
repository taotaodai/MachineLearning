# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns',None)
data_train = pd.read_csv("train.csv")

print(data_train.keys())


print(data_train[0:10])
"-----------------------特征---------------------"
"0.PassengerId"
"1.Survived： 1为幸存 0为死亡"
"2.Pclass(船舱等级)：1>2>3"
"3.Name"
"4.Sex：female>male"
"5.Age"
"6.SibSp(一同上船的兄弟姐妹或配偶)"
"7.Parch"
"8.Ticket"
"9.Fare(票价)"
"10.Cabin(船舱)"
"11.Embarked"

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind="bar")
plt.ylabel(u"获救情况 (1为获救)")
plt.title(u"人数")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title(u"按年龄看获救分布 (1为获救)")

plt.show()