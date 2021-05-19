# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:26:32 2020

@author: Shreya Biswas
"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import csv, os, sys
from sklearn.svm import SVC
import MFO as mf
import DA as da
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



def calc_acc(y, y_pred):
    total=len(y)
    c=0
    for i in range(0,total):
        if y[i]==y_pred[i]:
            c+=1
    return c/total
    

########### Original Feature set ###############
dataset=pd.read_csv('D:/Project/Thermal FER/Thermal_image_feature_3.csv') 
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

# x=x.to_numpy()
# y=y.to_list()

# c=1
# ch=y[0]

# y_ch=[]

# for i in range(0,len(y)):
#     if ch == y[i]:
#         y_ch.append(c)
#     else:
#         c+=1
#         ch=y[i]
#         y_ch.append(c)
# y_ch=np.asarray(y_ch)

######## Spliting into test and train ############
x_train_t,x_test_t, y_train_t, y_test_t=train_test_split(x,y,test_size=0.30)
svclassifier = SVC(kernel='poly',coef0=2)
#svclassifier = KNeighborsClassifier(n_neighbors=5)
#svclassifier = RandomForestClassifier()
svclassifier.fit(x_train_t, y_train_t)
y_test_tp=y_test_t.to_numpy()
y_pred = svclassifier.predict(x_test_t)
acc=calc_acc(y_test_tp,y_pred)
print("Original dimension = " + str(len(x.columns)))
# print()
print("accuracy = " + str(acc))

m=0
n=1
for i in range(1000):
    x_train_t,x_test_t, y_train_t, y_test_t=train_test_split(x,y,test_size=0.30)
    svclassifier = SVC(kernel='poly',coef0=2.0)
    #svclassifier = KNeighborsClassifier(n_neighbors=5)
    #svclassifier = RandomForestClassifier()
    svclassifier.fit(x_train_t, y_train_t)
    y_test_tp=y_test_t.to_numpy()
    y_pred = svclassifier.predict(x_test_t)
    acc=calc_acc(y_test_tp,y_pred)
    
    if m<acc:
        m=acc
        x_train,x_test, y_train, y_test= x_train_t,x_test_t, y_train_t, y_test_t
    if n>acc:
        n=acc

print(m)
print(n)   
orig=np.arange(len(x_train.iloc[0]))     

selected_features, fitness=mf.MFO(x_train, y_train, x_test, y_test, 100)
#selected_features, fitness, precision, sensitivity, F1, AUC=da.DA(x_train, y_train, x_test, y_test, 1, m, orig)
#selected_features = np.random.randint(288, size=)

x_train_selected_features=pd.DataFrame()
x_test_selected_features=pd.DataFrame()

reduced_dataset=pd.DataFrame()


for i in range(len(selected_features)):
    x_train_selected_features[str(selected_features[i])] = x_train.iloc[:,int(selected_features[i])] # X_Train from selected features
    x_test_selected_features[str(selected_features[i])] = x_test.iloc[:,int(selected_features[i])] # # X_Test from selected features
    reduced_dataset['attr ' + str(int(selected_features[i]))] = dataset.iloc[:,int(selected_features[i])]
    
reduced_dataset['class'] = dataset.iloc[:,-1]  


############ saving the the reduced feature set #####################

#reduced_dataset.to_csv('D:/Project/COVID-19/CTcovid/concat_layer_1/CTcovid_feature_2layerfusion_DA-FS.csv', index=False) 
    
#     x_train_selected_features[str(selected_features[i])] = x_train[:,selected_features[i]]
#     x_test_selected_features[str(selected_features[i])] = x_test[:,selected_features[i]]

# x_train_selected_features=x_train_selected_features.to_numpy()
# x_test_selected_features=x_test_selected_features.to_numpy()


# svclassifier = SVC(kernel='linear')
# svclassifier.fit(x_train_selected_features, y_train)
# y_test=y_test.to_numpy()
# y_pred = svclassifier.predict(x_test_selected_features)


# acc=calc_acc(y_test,y_pred)

print("Reduced dimension = " + str(len(np.unique(selected_features))))
# # print()
print("accuracy = " + str(fitness))
# print("precision = " + str(precision))
# print("sensitivity = " + str(sensitivity))
# print("AUC = " + str(AUC))
# print("F1 = " + str(F1))


# # print(acc)

x=reduced_dataset.iloc[:,:-1]
y=reduced_dataset.iloc[:,-1]
svclassifier = SVC(kernel='poly',coef0=2.0)
#svclassifier = KNeighborsClassifier(n_neighbors=5)
#svclassifier = RandomForestClassifier()
score_acc = cross_val_score(svclassifier,x, y, cv=5)
# score_prec= cross_val_score(svclassifier,x, y, cv=5, scoring='precision')
# score_rec= cross_val_score(svclassifier,x, y, cv=5, scoring='recall')
# score_f1= cross_val_score(svclassifier,x, y, cv=5, scoring='f1')
# score_auc= cross_val_score(svclassifier,x, y, cv=5, scoring='roc_auc')


print('cross val accuracy = ' + str(score_acc.mean()))
# print("cross val precision = " + str(score_prec.mean()))
# print("cross val sensitivity = " + str(score_rec.mean()))
# print("cross val AUC = " + str(score_auc.mean()))
# print("cross val F1 = " + str(score_f1.mean()))

