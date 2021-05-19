# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 01:08:18 2020

@author: Soumyajit Saha
"""


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import random
import math 
from sklearn.ensemble import RandomForestClassifier

###### Calculation of accuracy #####
def calc_acc(y, y_pred):
    total=len(y)
    c=0
    for i in range(0,total):
        if y[i]==y_pred[i]:
            c+=1
    return c/total

####### Calculation of fitness #########
def fitness(moth_pos, x_train, y_train, x_test, y_test):
    
    x_train_selected_features=pd.DataFrame()
    x_test_selected_features=pd.DataFrame()


    for i in range(len(moth_pos)):
       x_train_selected_features[str(moth_pos[i])] = x_train.iloc[:,int(moth_pos[i])]
       x_test_selected_features[str(moth_pos[i])] = x_test.iloc[:,int(moth_pos[i])]
    
    svclassifier = SVC(kernel='poly',coef0=2)
    #svclassifier=KNeighborsClassifier(n_neighbors=5)
    #svclassifier=RandomForestClassifier()
    svclassifier.fit(x_train_selected_features, y_train)
    y_test=y_test.to_numpy()
    y_pred = svclassifier.predict(x_test_selected_features)

    acc=calc_acc(y_test,y_pred)
    
    return acc
    
######### Moth Flame Algo ############
def MFO(x_train, y_train, x_test, y_test, Max_iteration):
    
    length=len(x_train.iloc[0])
    size1=int(length)
    
    moth_population=np.zeros(shape=(20,size1))
    fitness_of_moths = np.zeros(20)

    ########## Initialisation of solutions randomly ##############
    for i in range(0,20):
            # for j in range(0,size):
            #     if j==0:
            #         r=random.randint(0,length-1)
            #         while( (length-1-r)<(size-j-1) ):
            #             r=random.randint(0,length-1)
            #         moth_population[i][j]=r
            #     else:
            #         r=random.randint(0,length-1)
            #         while( ((length-1-r)<(size-j-1) ) or moth_population[i][j-1]>=r):
            #             r=random.randint(0,length-1)
                    # moth_population[i][j]=r
                    
        moth_population[i]=np.random.randint(length, size=size1)
    previous_population=0
    previous_fitness=0
    
    # Max_iteration=m  
    ############ Begin of Algo ##############
    for Iteration in range(1,Max_iteration+1):
        
        
        print(str(Iteration), end=" ")
        Flame_no = round(20-Iteration*((20-1)/Max_iteration));
        
        for i in range(0,20):
            fitness_of_moths[i] = fitness(moth_population[i],x_train, y_train, x_test, y_test) # calculation of fitness
        
        
        if Iteration==1:
             best_flame_fitness = np.sort(fitness_of_moths)[::-1] # getting best flames
             I = np.argsort(fitness_of_moths)[::-1]
             best_flames = moth_population[I]
            
        else:
            double_population = np.concatenate((previous_population,best_flames))
            double_fitness=np.concatenate((previous_fitness, best_flame_fitness))
            
            double_fitness_sorted = np.sort(double_fitness)[::-1]
            I=np.argsort(double_fitness)[::-1]
            double_sorted_population = double_population[I]
            
            best_flame_fitness = double_fitness_sorted[0:20] # getting best flames
            best_flames = double_sorted_population[0:20]
            
        Best_flame_score=best_flame_fitness[1] # the best flame
        Best_flame_pos=best_flames[1]
      
        previous_population=moth_population
        previous_fitness=fitness_of_moths
            
        a=-1 + Iteration*((-1)/Max_iteration);   
        
        for i in range(0,20):
            for j in range(0,size1):
                
                if i<=Flame_no:
                    distance_to_flame = abs(best_flames[i][j] - moth_population[i][j])
                    b = 1
                    t = (a-1)*random.random() + 1
                    moth_population[i][j] = round(distance_to_flame*math.exp(b*t)*math.cos(t*2*math.pi) + best_flames[i][j]) # updating moth population
                    
                    if moth_population[i][j]<0: # Bringinging back to search space
                        moth_population[i][j]=0
                    
                    if moth_population[i][j]>length-1: # Bringinging back to search space
                        moth_population[i][j]=length-1
                    
                else:
                    distance_to_flame = abs(best_flames[i][j] - moth_population[i][j])
                    b = 1
                    t = (a-1)*random.random() + 1
                    moth_population[i][j] = round(distance_to_flame*math.exp(b*t)*math.cos(t*2*math.pi) + best_flames[Flame_no][j]) # updating moth population
                    
                    if moth_population[i][j]<0: # Bringinging back to search space
                        moth_population[i][j]=0
                        
                    if moth_population[i][j]>length-1: # Bringinging back to search space
                        moth_population[i][j]=length-1
    
        #print("Iteration = " + str(Iteration))
        #print()
        
    return Best_flame_pos, Best_flame_score