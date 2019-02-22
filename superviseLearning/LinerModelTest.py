
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


cancer = load_breast_cancer()

X_train,X_test,y_train,y_test = train_test_split(cancer['data'],cancer['target'],stratify=cancer['target'],random_state=66)

clf_1 = LogisticRegression().fit(X_train,y_train)
#print("LogisticRegression：{}".format(clf.score(X_train,y_train)))

print("LogisticRegression：{}".format(clf_1.score(X_test,y_test)))

clf_2 = LinearSVC().fit(X_train,y_train)

print("LogisticRegression：{}".format(clf_2.score(X_test,y_test)))