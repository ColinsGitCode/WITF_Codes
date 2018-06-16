import numpy.matlib
import numpy as np
import scipy 

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

def Multipy(mat1,mat2):
    product = np.dot(mat1, mat2)
    return product



