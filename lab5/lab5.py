# -*- coding: utf-8 -*-
from cmath import log
import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy
import sys
sys.path.append("/Users/peipi98/Documents/PoliTO/Materie/Machine Learning/labs/MachineLearning_PatternRecognition/functions")
from mlFunc import *

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) 
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)

if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    h = {}
    
    for i in range(3):
        mu, C = ML_GAU(DTR[:, LTR==i])
        h[i] = (mu, C)
        
    SJoint = numpy.zeros((3, DTE.shape[1]))
    logSJoint = numpy.zeros((3, DTE.shape[1]))
    classPriors = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    
    for label in range(3):
        mu, C = h[label]
        
        SJoint[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, C).ravel()) * classPriors[label] 
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + numpy.log(classPriors[label])
        
    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)
    
    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)
    
    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    
    accuracy_1 = (LTE == LPred1).sum() /LTE.size
    error_1 = 1 - accuracy_1
    
    accuracy_2 = (LTE == LPred2).sum() /LTE.size
    error_2 = 1 - accuracy_2