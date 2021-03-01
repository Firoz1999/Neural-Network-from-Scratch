#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:55:21 2020

@author: firoz
"""
'''
0.881003	0.0933232	0.864105	0.859712	0.630303
0.567115	0.0520928	0.29044	0.294015	0.826395
0.0731895	0.241731	0.625217	0.270973	0.368328
0.716521	0.518325	0.219095	0.0558455	0.22474

0.454197	0.203009	0.388456	0.978316	0.451603
0.709381	0.56044	0.248393	0.72212	0.0367601
0.125562	0.170162	0.715046	0.33438	0.52043

'''

import numpy as np
import pandas as pd
from random import random

dataset = pd.read_csv('iris.csv')
Set=dataset.iloc[0:50,:4].values
Vers=dataset.iloc[50:100,:4].values
Virg=dataset.iloc[100:150,:4].values

X=np.vstack((Set,Vers,Virg))
X=np.insert(X,0,1,axis=1)
yc=dataset.iloc[:,[4]].values
yy=[]
for item in yc:
    print(item)
    if(item=='Iris-setosa'):
        yy.append([1,0,0])
    if(item=='Iris-versicolor'):
        yy.append([0,1,0])
    if(item=='Iris-virginica'):
        yy.append([0,0,1])
y=np.array(yy)*1.
w1=[]
w2=[]
for i in range(4):
    #np.random.seed(1)
    w1.append([random(),random(),random(),random(),random()])
    
for i in range(3):
    w2.append([random(),random(),random(),random(),random()])
w1=np.array(w1)
w2=np.array(w2)
learning_rate=0.1
#h0=X
def sigmoid(t):
    return 1/(1+np.exp(-t))
    #return (2/(1+np.exp(-2*t)))-1

def sigmoid_derivative(t):
    return t*(1-t)
    #return 1-t*t

class NeuralNetwork:
    def __init__(self,X,y,w1,w2,learning_rate):
        self.X=X
        self.y=y
        self.w1=w1
        self.w2=w2
        self.learning_rate=learning_rate
        
    def feedforward(self,x):
        self.h0=x
        self.h1=sigmoid(np.dot(self.w1,self.h0.T))
        self.h2=sigmoid(np.dot(self.w2,np.insert(self.h1,0,1,axis=0)))
        return self.h2
        
    def backpropagate(self,y):
        #gg=sigmoid_derivative(h2)
        #print("y : ",y)
        delta_2=(self.h2-y.T)*sigmoid_derivative(self.h2)
        h1_temp=np.insert(self.h1,0,1,axis=0).T
        dE_dw2=np.dot(delta_2,h1_temp)
        
        delta_1=sigmoid_derivative(self.h1)*np.dot(delta_2.T,w2[:,1:]).T
        dE_dw1=np.dot(delta_1,self.h0)
        
        self.w2=self.w2-learning_rate*dE_dw2
        self.w1=self.w1-learning_rate*dE_dw1
        
    def train(self,x,y):
        actual_output=self.feedforward(x)
        self.backpropagate(y)
        return actual_output

NN=NeuralNetwork(X,y,w1,w2,learning_rate)

itr=1
ans=[]
correct=0
while(itr<=10000):
    for i in range(len(X)):
        xi=[]
        xi.append(X[i])
        xi=np.array(xi)
        yi=[]
        yi.append(y[i])
        yi=np.array(yi)
        #print(xi,yi)
        NN.train(xi,yi)
        #print("iteration : ",itr)
        temp=NN.feedforward(xi)
        #print("Predicted : \n",temp)
        ans.append(temp)
        #print("Loss : \n",np.square(yi-NN.feedforward(xi).T))
        if(yc[i]=='Iris-setosa'):
            if(temp[0][0]>=0.5):
                correct+=1
        if(yc[i]=='Iris-versicolor'):
            if(temp[1][0]>0.5):
                correct+=1
        if(yc[i]=='Iris-virginica'):
            if(temp[2][0]>0.5):
                correct+=1
    print("itr : ",itr)
    print("\nCrrect : ",correct/150)
    correct=0
        
    itr+=1
#print(NN.train())
    
