# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:38:48 2018

@author: tssaib01
"""



from sklearn.datasets import load_iris

from sklearn.cross_validation import train_test_split



from sklearn import metrics

def data_load():
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.4,random_state=4)
    return X_train,X_test,Y_train,Y_test


# This is a code to fit the Logistic regression model

from sklearn.linear_model import LogisticRegression

X_train,X_test,Y_train,Y_test=data_load()

logreg=LogisticRegression()

logreg.fit(X_train,Y_train)


Y_predict=logreg.predict(X_test)

print(metrics.accuracy_score(Y_test,Y_predict))

#0.95  Accuracy score

#This is the code to fir the KNN Model



from sklearn.neighbors import KNeighborsClassifier


X_train,X_test,Y_train,Y_test=data_load()

KNN=KNeighborsClassifier()

KNN.fit(X_train,Y_train)

Y_predict=KNN.predict(X_test)

print(metrics.accuracy_score(Y_test,Y_predict))

#0.967 Accuracy Score


#This is the code to fit Naive Bayes algorithm


from sklearn.naive_bayes import BernoulliNB


X_train,X_test,Y_train,Y_test=data_load()
Bernoulli=BernoulliNB()

Bernoulli.fit(X_train,Y_train)

Y_predict=Bernoulli.predict(X_test)

print(metrics.accuracy_score(Y_test,Y_predict))


#0.2833
