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
import scipy.linalg

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

# aim: dimensionality reduction of a dataset with PCA
def PCA(D):

# 1. compute covariance matrix
    n = numpy.shape(D)[1]
    # mu = dataset mean, calculated by axis = 1 (columns mean)
    # the result is an array of means for each column
    mu = D.mean(1)

    # remove the mean from all points of the data matrix D,
    # so I can center the data
    DC = D - mcol(mu)

    #calculate covariance matrix with DataCentered matrix
    C = 1/n * numpy.dot(DC, numpy.transpose(DC))

    #Calculate eigenvectors and eigenvalues of C with singular value decomposition
    # That's why C is semi-definite positive, so we can get the sorted eigenvectors
    # from the svd: C=U*(Sigma)*V(^transposed)
    # svd() returns sorted eigenvalues from smallest to largest,
    # and the corresponding eigenvectors
    USVD, s, _ = numpy.linalg.svd(C)

    # m are the leading eigenvectors chosen from the next P matrix
    m = 2
    P = USVD[:, 0:m]

    #apply the projection to the matrix of samples D
    DP = numpy.dot(P.T, D)
    print(DP)
    hlabels = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    for i in range(3):
#I have to invert the sign of the second eigenvector to flip the image
        plt.scatter(DP[:, L==i][0], -DP[:, L==i][1], label = hlabels.get(i))
        plt.legend()
        plt.tight_layout()
    plt.show()
    
def LDA(D, L):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]
    DPn = []

# To compute LDA, we have to:
# 1. Compute matrices SB and SW
# 2. Compute LDA directions 
# 3. Solving the eigenvalue problem by joint diagonalization of Sb and Sw

# ************** 1. Compute matrices SB and SW **************
    # tot. number of samples
    N = numpy.shape(D)[1]

# compute S_{w,c}
    tot_Swc = 0
    muc_vec = []
    nc_vec = []
    for i in range(3):
        Dtemp = D[:, L==i]
        nc = numpy.shape(Dtemp)[1]
        nc_vec.append(nc)
        muc = Dtemp.mean(1)
        muc_vec.append(muc)
        DC = Dtemp - mcol(muc)
# since we have to implement the sum of within covariance of all classes,
# we have to calculate a covariance matrix step by step but with the substitution
# of 1/nc with nc to fit in the general formula in tot_Swc 
        C = nc * numpy.dot(DC, numpy.transpose(DC))
        USVD, s, _ = numpy.linalg.svd(C)
        m = 2
        P = USVD[:, 0:m]
        DP = numpy.dot(P.T, D)
        DPn.append(DP)
        tot_Swc += C * 1/nc
    Sw = tot_Swc / N

# Compute S_B
    # compute generic mean vector
    mu = D.mean(1)
    tot_SB = 0
    for i in range(3):
        tot_SB += nc_vec[i] * mcol((muc_vec[i] - mu)).dot(mcol(muc_vec[i] - mu).T)

    Sb = 1/N * tot_SB

# ************** 2. Compute LDA directions **************
    # solve the eigenvalue problem for hermitian matrices with scipy.linalg.eigh
    # Pay attention: numpy.linalg.eigh does not resolve this problem.
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    # Since W column are not necessarily orthogonal, we can find a basis U
    # for the subspace spanned by W using the svd:
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]
    print(U)
# ************** 3. Solving the eigenvalue problem 
# by joint diagonalization of Sb and Sw **************
# We have to estimate the matrix P1 such that the within class covariance
# of the transformed points P1x is the identity (see pdf)
    U, s, _ = numpy.linalg.svd(Sw)

    # s is the diagonal of the matrix Sigma.
    P1 = numpy.dot(U * numpy.diag(1.0/(s**0.5)), U.T)

    #now we have to calculate Sbt as follows to apply
    # the whitening transformation as follows:
    Sbt = numpy.dot(numpy.dot(P1,Sb), P1.T)

    U, s, _ = numpy.linalg.svd(Sbt)
    P2 = numpy.dot(U * numpy.diag(1.0/(s**0.5)), U.T)

    W = numpy.dot(P1.T, P2)
    print(numpy.shape(W))
    hlabels = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
    for i in range(3):
#I have to invert the sign of the second eigenvector to flip the image
        plt.scatter(U[:, L==i][0], -U[:, L==i][1], label = hlabels.get(i))
        plt.legend()
        plt.tight_layout()
    plt.show()
    



if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=10)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    D, L = load('iris.csv')
    print(L)
    PCA(D)
    LDA(D,L)

