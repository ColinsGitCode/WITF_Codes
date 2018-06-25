# Python3
# Filename: TensorIrr.py
# Usages : Construct the Irregular Tensor

import numpy as np
import scipy

from SaveData import *

def CreateDefaultSparseDokMatrix(row,col):
    sm = scipy.sparse.dok_matrix((row,col), dtype=np.float16)
    for i in range(row):
        for j in range(col):
            if i == j:
                sm[i,j] = i + j

    return sm

def CreateSparseDokMatrix(row,col):
    sm = scipy.sparse.dok_matrix((row,col), dtype=np.float16)
    for i in range(row):
        for j in range(col):
            if i == (9-j):
                sm[i,j] = i + j
    return sm                

