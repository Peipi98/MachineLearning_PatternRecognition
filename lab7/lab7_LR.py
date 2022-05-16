#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:46:34 2022

@author: peipi98
"""
from cmath import log
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy
import scipy.optimize as opt
import sklearn.datasets
import sys
sys.path.append("/Users/peipi98/Documents/PoliTO/Materie/Machine Learning/labs/MachineLearning_PatternRecognition/functions")
from mlFunc import *

def f(x):
    return pow(x[0] + 3, 2) + np.sin(x[0]) + pow(x[1] + 1, 2)

def f_2(x):
    y,z = x;
    
    obj = pow(x[0] + 3, 2) + np.sin(x[0]) + pow(x[1] + 1, 2)
    grad = np.array([2*(y+3) + np.cos(y), 2*(z+1)])
    
    return obj, grad

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2) 
    return D, L

def logreg_obj_wrap(DTR, LTR, 1):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v):
        w, b = v[0:M], v[-1]
        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(, -S*Z).mean()
        
        return cxe + 0.5*l * np.linalg.norm(w)**2
    return logreg_obj

if __name__ == "__main__":
    
    #(x1, f1, d1) = opt.fmin_l_bfgs_b(f, mcol(np.array([0,0])), approx_grad=True, iprint=1)
    
    #(x2, f2, d2) = opt.fmin_l_bfgs_b(f_2, np.zeros(2), iprint=1)
    
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    
    
    