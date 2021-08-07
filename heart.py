import pandas as pd
import numpy as np

import seaborn as sns

df = pd.read_csv("heart.csv")

df.shape

df.head(10)

df.info()


df["target"].unique()

df.describe()



df.isnull().sum()

df.head(50).plot(kind='area',figsize=(10,5))

df.plot(x='age',y='cp',kind='scatter',figsize =(10,10))


df["target"].describe()


df["target"].unique()

print(df.corr()["target"].abs().sort_values(ascending=False))

y = df["target"]

sns.countplot(y)


target_temp = df.target.value_counts()

print(target_temp)

df["sex"].unique()

sns.barplot(df["sex"],y)

df["slope"].unique()

sns.barplot(df["slope"],y)

from sklearn.model_selection import train_test_split

predictors = df.drop("target",axis=1)
target = df["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

X_train.shape


X_test.shape


Y_train.shape


Y_test.shape

from sklearn.metrics import accuracy_score

from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)


Y_pred_svm.shape


score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using  SVM is: "+str(score_svm)+" %")

