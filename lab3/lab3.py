#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:45:50 2022

@author: peipi98
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab

def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((1, v.size))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
#hstack(DList) crea una matrice 4x150 in cui ogni riga rappresenta un attributo,
#mentre le colonne i valori

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def dim_reduction(D):
    n = numpy.shape(D)[1]
    mu = D.mean(1)
    DC = D - mcol(mu)

    C = 1/n * numpy.dot(DC, numpy.transpose(DC))

    USVD, s, _ = numpy.linalg.svd(C)
    print("USVD")
    print(USVD)
    m = 2
    P = USVD[:, 0:m]

    DP = numpy.dot(P.T, D)
    print(DP)
    print(DP.shape)
    pylab.scatter(DP[0], DP[1])
    print(DP[0])
    plt.scatter(DP[0], DP[1])
    plt.show()
    
if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('iris.csv')
    print(numpy.shape(D))
    dim_reduction(D)
    #plot_hist(D, L)
    #plot_scatter(D, L)

