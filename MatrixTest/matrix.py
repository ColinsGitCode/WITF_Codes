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
