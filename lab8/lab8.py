import sys
import numpy as np
from prettytable import PrettyTable
sys.path.append("./functions")
from mlFunc import *

def confusion(ll_0, ll_1, ll_2, n):
    t = PrettyTable(["", "0", "1", "2"])
    t.title = "Conf. matrix for iris dataset"



if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    m = mu(DTR)
    C = covariance(DTR, m)
    n_0 = numpy.shape(DTR[:, LTR==0])[1]
    n_1 = numpy.shape(DTR[:, LTR==1])[1]
    n_2 = numpy.shape(DTR[:, LTR==2])[1]
    print(n_0)
    ll_0 = loglikelihood(DTR[:, LTR==0], m, C)
    ll_1 = loglikelihood(DTR[:, LTR==1], m, C)
    ll_2 = loglikelihood(DTR[:, LTR==2], m, C)
    print(ll_0)
    print(ll_1)
    print(ll_2)


    