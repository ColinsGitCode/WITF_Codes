import numpy.matlib
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph

# -----------------------------------

def test():
    a = np.array([[1,2],[3,4]])
    b = np.array([[11,12],[13,14]])
    result = np.dot(a,b)
    return result

def main():
    print(test())

def CreateMat(row,col):
    mat = np.matlib.rand(row,col)
    return mat

def NumpyMultipy(mat1,mat2):
    product = np.dot(mat1, mat2)
    return product

def CooRandMat(row,col,dense=0.25):
    mat = sparse.random(row,col,density=dense)
    return mat

def Return2Mats(row,col,dense=0.25):
    m1 = CooRandMat(row,col,dense)
    m2 = m1.T
    return m1,m2

def Type2Mats(row,col,dense=0.25,type='coo'):
    print("fromat is %s!" %type )
    m1 = sparse.random(row,col,density=dense,format='dok')
    m2 = m1.T
    return m1,m2


