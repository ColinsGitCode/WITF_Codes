# Python3
# Filename: TensorIrr.py
# Usages : Construct the Irregular Tensor

import numpy as np
import scipy

from SaveData import *

class TensorIrr:
    ''' Class for the Irregular Tensor '''
    
    def __init__(self):
        '''
        __init__ function to load the txt data files 
        '''
        self.two_more_ratings_users_dic = \
        LoadFromTxt("TwoMoreRatingsUsers.txt")
        self.userIDs_pos = LoadFromTxt("TwoRatingsUsersPosMap.txt")
        self.selected_five_category = LoadFromTxt("FiveSelectdCategories.txt")
        self.sparse_matrix_dic = {}

    def init_sparse_matrix(self):
        '''
        To init the empty sparse matrices 
        '''
        users_conuts = len(self.userIDs_pos)
        for cate in self.selected_five_category:
            items_counts = len(self.selected_five_category[cate][1])
            sm = scipy.sparse.dok_matrix((users_conuts,item_counts),
                    dtype=np.int16)
            self.sparse_matrix_dic[cate] = sm
            print("matrix for category: %d, size is%d and %d!" %(cate,users_conuts,item_counts))
        return True



# ------------------------------------------------------------------------------
# Other Parts
# ------------------------------------------------------------------------------

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
# -----------------------------------------------------------------------------
# main parts
# -----------------------------------------------------------------------------
tensor = TensorIrr()
