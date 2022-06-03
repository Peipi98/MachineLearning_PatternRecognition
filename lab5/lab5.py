# -*- coding: utf-8 -*-
from cmath import log
from ctypes.wintypes import LPRECT
import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy
import sys
sys.path.append("./functions")
from mlFunc import *

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    print(idx)
    print(idx.shape)
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)

def kfold_cross(D, L, k, seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    numpy.random.seed(seed)
    
    idx = numpy.random.permutation(D.shape[1])
    idx = numpy.array(idx)
    #print(idx.shape)
    idx = numpy.reshape(idx, (k, idx.shape[0]//k))
    
    return idx

def evaluate_MVG(D, L):
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    h = {}
    
    for i in range(3):
        mu, C = ML_GAU(DTR[:, LTR==i])
        # for Naive Bayes assumption deccoment:
        # C = numpy.diag(numpy.diag(C))
        
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
    return LPred1, LPred2

def evaluate_tied_GC(D, L):
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # compute tied covariance matrix
    N = numpy.shape(DTR)[1]
    Ctied = numpy.zeros((4,4))
    
    h = {}
    
    for i in range(3):
        _, Ctemp = ML_GAU(DTR[:, LTR==i])
        Nc = numpy.shape(DTR[:, LTR==i])[1]
        # for Naive Bayes assumption deccoment:
        # C = numpy.diag(numpy.diag(C))
        Ctied += Ctemp * Nc
    Ctied /= N
    
    for i in range(3):
        mu , _ = ML_GAU(DTR[:, LTR==i])
        h[i] = (mu, Ctied)
        
    
    SJoint_tied = numpy.zeros((3, DTE.shape[1]))
    logSJoint_tied = numpy.zeros((3, DTE.shape[1]))
    classPriors_tied = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    
    for label in range(3):
        mu, C = h[label]
        
        SJoint_tied[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, C).ravel()) * classPriors_tied[label] 
        logSJoint_tied[label, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + numpy.log(classPriors_tied[label])
        
    SMarginal = SJoint_tied.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint_tied, axis=0)
    
    Post1 = SJoint_tied / mrow(SMarginal)
    logPost = logSJoint_tied - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)
    
    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    
    accuracy_1 = (LTE == LPred1).sum() /LTE.size
    error_1 = 1 - accuracy_1
    print(accuracy_1)
    accuracy_2 = (LTE == LPred2).sum() /LTE.size
    error_2 = 1 - accuracy_2
        
if __name__ == "__main__":
    D, L = load_iris()
    
    LPred1, LPred2 = evaluate_MVG(D, L)
    print(LPred1)
    print(LPred2)
    #evaluate_tied_GC(D, L)
    print(kfold_cross(D, L, 3))