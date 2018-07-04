# ----------------------------------------------------------------------------
# <------------------------- Start of the Codes Zone ------------------------>
#                            Zone Title

#                            Zone Title
# <------------------------- End of the Codes Zone -------------------------->
# ----------------------------------------------------------------------------
import numpy.matlib
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph
import timeit

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# <------------------------- Start of the Codes Zone ------------------------>
#                            Zone Title : Dense Matrix Parts

def CreateDenseMat(row,col):
    mat = np.matlib.rand(row,col)
    return mat

def NumpyMultipy(mat1,mat2):
    product = np.dot(mat1, mat2)
    return product

#                            Zone Title: Dense Matrix Parts
# <------------------------- End of the Codes Zone -------------------------->
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# <------------------------- Start of the Codes Zone ------------------------>
#                            Zone Title: Sparse Matrix Parts

def CooRandSparseMat(row,col,dense=0.25):
    mat = sparse.random(row,col,density=dense)
    return mat

def Return2CooSparseMats(row,col,dense=0.25):
    m1 = CooRandMat(row,col,dense)
    m2 = m1.T
    return m1,m2

def Create2FormatSparseMats(row,col,dense=0.25,type='coo'):
    print("fromat is %s!" %type )
    m1 = sparse.random(row,col,density=dense,format='dok')
    m2 = m1.T
    return m1,m2

#                            Zone Title: Sparse Matrix Parts
# <------------------------- End of the Codes Zone -------------------------->
# ----------------------------------------------------------------------------


def get_dok_sparse_mats(row,col,dense=0.004):
    m1 = sparse.random(row,col,density=dense,format='dok')
    m2 = m1.tocoo()
    return m1,m2

# ------------------------------------------------------------------

def func1(x):
    pow(x, 2)

def func2(x):
    return x * x

def dot_time(mat,mat_T):
    return mat.dot(mat_T)

def transpose_time(mat):
    return mat.transpose()

def Hadamard_time(mat):
    return mat.multiply(mat)



v = 10000

func1_test = 'func1(' + str(v) + ')'
func2_test = 'func2(' + str(v) + ')'
print(timeit.timeit(func1_test, 'from __main__ import func1'))
print(timeit.timeit(func2_test, 'from __main__ import func2'))

print(timeit.repeat(func1_test, 'from __main__ import func1'))
print(timeit.repeat(func2_test, 'from __main__ import func2'))


