# Python3
# Filename: TensorIrr.py
# Usages : Construct the Irregular Tensor

import numpy as np
import scipy
from scipy.sparse import dok_matrix

from SaveData import *

class TensorIrr:
    ''' Class for the Irregular Tensor '''
    
    def __init__(self):
        '''
        __init__ function to load the txt data files 
        '''
        self.two_more_ratings_users_dic = \
        load_from_txt("TwoMoreRatingsUsers.txt")
        self.userIDs_pos = load_from_txt("TwoRatingsUsersPosMap.txt")
        self.selected_five_category = load_from_txt("FiveSelectdCategories.txt")
        self.sparse_matrix_dic = {}

    def init_sparse_matrix(self):
        '''
        To init the empty sparse matrices 
        '''
        users_conuts = len(self.userIDs_pos)
        for cate in self.selected_five_category:
            items_counts = len(self.selected_five_category[cate][1])
            sm = dok_matrix((users_conuts,items_counts),
                    dtype=np.int16)
            self.sparse_matrix_dic[cate] = sm
            print("matrix for category: %d, size is%d and %d!"
                    %(cate,users_conuts,items_counts))
        return True

    def update_sparse_matrix(self):
        '''
        update sparse matrix
        '''
        # 遍历所有的用户, user_pos 为该用户在矩阵的行位置
        for user_pos in range(len(self.userIDs_pos)):
            userID = self.userIDs_pos[user_pos]
            user_ratings_dic = self.two_more_ratings_users_dic[userID]
            # 遍历此用户在所有类目下的评分，数据结构为字典{cateID:[(itemID,ratings),.....]}
            for user_cateID in user_ratings_dic:
                user_cateID_ratings_list = user_ratings_dic[user_cateID]
                # 遍历此用户在此类目下所有的评分，评分为(itemID,rating)
                for rating in user_cateID_ratings_list:
                    itemID = rating[0]
                    rating_value = rating[1]
                    items_cateID_npa = \
                    self.selected_five_category[user_cateID][1]
                    # item_pos : 该item在矩阵中列位置
                    item_pos = np.where(items_cateID_npa == itemID) 
                    item_pos = item_pos[0][0]
                    self.sparse_matrix_dic[user_cateID][user_pos,item_pos] = \
                    rating_value
                    print("Update the rating which category:%d, UserID:%d, \
                    UserPos:%d,ItemID:%d, ItemPos:%d !" \
                    % (user_cateID, userID, user_pos, itemID, item_pos))
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
