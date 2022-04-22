from cmath import log
import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sys
sys.path.append("/Users/peipi98/Documents/PoliTO/Materie/Machine Learning/labs/MachineLearning_PatternRecognition/functions")
from mlFunc import *

def logpdf_1sample(x,mu,C):
    P = numpy.linalg.inv(C)
    res = -0.5 * x.shape[0] * numpy.log(2*numpy.pi)
    res += -0.5 * numpy.linalg.slogdet(C)[1]
    res += -0.5 * numpy.dot((x-mu).T, numpy.dot(P, (x-mu)))
    
    return res.ravel()

def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_1sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(Y).ravel()

def logpdf_GAU_ND_Opt(X, mu, C):
    P  = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]

    Y = []

    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * numpy.dot((x-mu).T, numpy.dot(P, (x-mu)))
        Y.append(res)
    return numpy.array(Y).ravel()

def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND_Opt(XND, m_ML, C_ML).sum()

def plot():
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000) 
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND_Opt(mrow(XPlot), m, C))) 
    plt.show()

def plot_hist_exp(X1D, m_ML, C_ML):
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m_ML, C_ML)))
    plt.show()

if __name__ == "__main__":
    D, L = load("/Users/peipi98/Documents/PoliTO/Materie/Machine Learning/labs/MachineLearning_PatternRecognition/Lab3/iris.csv")
    #PCA(D, L)
    mu = mu(D)
    C = covariance(D, mu)
    prova = logpdf_GAU_ND(D, mu, C)
    prova_opt = logpdf_GAU_ND_Opt(D, mu, C)
    
    XND = numpy.load("XND.npy")
    muML = mcol(XND.mean(1))
    CML = covariance(XND, muML)

    ll = loglikelihood(XND, muML, CML)
    print(ll)

    X1D = numpy.load("X1D.npy")
    m_ML = mcol(X1D.mean(1))
    C_ML = covariance(X1D, m_ML)
    plot_hist_exp(X1D, m_ML, C_ML)