from ctypes.wintypes import LPRECT
import sys
import numpy as np
from prettytable import PrettyTable
sys.path.append("../functions")
from mlFunc import *
from classifiers import *

def confusion(LTE, LPred, c, title=None, header = None):
    t = PrettyTable(header)
    if title:
        t.title = title
    m = numpy.zeros((c,c))
    for i in range(c):
        row = [i]
        for j in range(c):
            m[i][j] = ((LPred == i) * (LTE == j)).sum()
            row.append(m[i][j])
        t.add_row(row)
    print(t)
    return m

def threshold(C, pi):
    t = -(numpy.log(pi * C[0, 1]) - numpy.log((1-pi)*C[1, 0]))
    return t

def opt_bayes(pi, Cfn, Cfp):
    pi = pi
    C = numpy.matrix([[0,Cfn],[Cfp,0]])
    t = threshold(C, pi)
    LPred = llr > t
    confusion(labels, LPred, 2, f"π={pi} - Cfn={Cfn}, Cfp={Cfp}", ["", "0", "1"])

if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
     

    LPred1, LPred2 = MGC(DTE, DTR, LTR, 3)
    confusion(LTE, LPred1, 3, "Conf. matrix - MVG", ["", "0", "1", "2"]) 

    _, LPred2 = tied_cov_GC(DTE, DTR, LTR, 3)
    confusion(LTE, LPred2, 3, "Conf. matrix - Tied MVG", ["", "0", "1", "2"])
    
    llr = numpy.load('commedia_llr_infpar.npy')
    labels = numpy.load("commedia_labels_infpar.npy")

    # =========== Optimal Bayes Decision ===========
    print("\n=========== Optimal Bayes Decision ===========\n")

    opt_bayes(0.5, 1, 1)
    opt_bayes(0.8, 1, 1)
    opt_bayes(0.5, 10, 1)
    opt_bayes(0.8, 1, 10)