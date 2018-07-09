# Python3
# Filename: TensorIrr.py
# Usages : Construct the Irregular Tensor

import numpy as np
import scipy
from scipy.sparse import dok_matrix

from SaveData import *
from functionDrafts import *

class TensorIrr:
    ''' Class for the Irregular Tensor '''
    
    def __init__(self):
        '''
        __init__ function to load the txt data files 
        '''
        # 每个用户在某个分类有两个以上评分就保留这个用户
        # DS : self.two_more_ratings_users_dic =
        #            { userID : { cateID : [(itemID1,rating1), (itemID2,rating2), ...], ....}, ... } 
        self.two_more_ratings_users_dic = \
        load_from_txt("/home/Colin/txtData/TwoMoreRatingsUsers.txt")
        # 任意分类有两个以上分类的用户ID的排序列表
        # DS : self.userIDs_pos = [userID1, userID2, ...] (sorted)
        self.userIDs_pos = load_from_txt("/home/Colin/txtData/TwoRatingsUsersPosMap.txt")
        # 选择的五个分类的所有的itemID, 保存在nparray中
        # DS : self.selected_five_category = 
        #            { cateID : ("cate_name" , ndarray[itemID,itemID,...](sorted)), ... }
        self.selected_five_category = load_from_txt("/home/Colin/txtData/FiveSelectdCategories.txt")
        # 每一个分类的稀疏矩阵，行坐标为用户在用户ID排序列表中的Index
        # 列坐标为item在Item列表(np.array)中的Index
        self.sparse_matrix_dic = {}
        # 在所有分类都有2个以上评分的用户
        # DS : self.user_has_2more_ratings_in_all_categores = 
        #            { same as self.two_more_ratings_users_dic }
        self.user_has_2more_ratings_in_all_categores = {}
        # 不满足条件的用户
        # DS : self.user_not_has_2more_ratings_in_all_categores = 
        #            { same as self.two_more_ratings_users_dic }
        self.user_not_has_2more_ratings_in_all_categores = {}
        # 在所有分类都有2个以上评分的用户的ID排序列表
        # DS : self.userIDs_pos_10ratings = [userID1, userID2, ...] (sorted)
        self.userIDs_pos_10ratings = []
        # 包含所有数据（稀疏矩阵，用户排序列表，item排序列表）的字典
        self.WITF_raw_data = {}
        # 保存每个用户的空白(blank_postions)位置
        # DS : self.userPos_blank_itemPos = 
        #              {userPos : {cateID : [blankItemsPos1, blankItemsPos2, ...](sorted), ...}, ... }
        self.userPos_blank_itemPos = {}

    def init_sparse_matrix(self):
        '''
        To init the empty sparse matrices 
        '''
        #users_conuts = len(self.userIDs_pos)  # 以前的用户选择标准
        users_conuts = len(self.userIDs_pos_10ratings)
        for cate in self.selected_five_category:
            items_counts = len(self.selected_five_category[cate][1])
            sm = dok_matrix((users_conuts,items_counts),
                    dtype=np.float16)
            self.sparse_matrix_dic[cate] = sm
            print("matrix for category: %d, size is%d and %d!"
                    %(cate,users_conuts,items_counts))
        return True

    def init_sparse_matrix_from_txtData(self):
        '''
        To init the empty sparse matrices 
        '''
        users_conuts = len(self.userIDs_pos)  # 以前的用户选择标准
        #users_conuts = len(self.userIDs_pos_10ratings)
        for cate in self.selected_five_category:
            items_counts = len(self.selected_five_category[cate][1])
            sm = dok_matrix((users_conuts,items_counts),
                    dtype=np.float16)
            self.sparse_matrix_dic[cate] = sm
            print("matrix for category: %d, size is%d and %d!"
                    %(cate,users_conuts,items_counts))
        return True

    def update_sparse_matrix(self):
        '''
        update sparse matrix
        '''
        # 遍历所有的用户, user_pos 为该用户在矩阵的行位置
        #for user_pos in range(len(self.userIDs_pos)):
        for user_pos in range(len(self.userIDs_pos_10ratings)):
            userID = self.userIDs_pos_10ratings[user_pos]
            #user_ratings_dic = self.two_more_ratings_users_dic[userID]
            user_ratings_dic = self.user_has_2more_ratings_in_all_categores[userID]
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

    def update_sparse_matrix_from_txtData(self):
        '''
        update sparse matrix
        '''
        # 遍历所有的用户, user_pos 为该用户在矩阵的行位置
        for user_pos in range(len(self.userIDs_pos)):
        #for user_pos in range(len(self.userIDs_pos_10ratings)):
            self.userPos_blank_itemPos[user_pos] = {}
            userID = self.userIDs_pos[user_pos]
            user_ratings_dic = self.two_more_ratings_users_dic[userID]
            #user_ratings_dic = self.user_has_2more_ratings_in_all_categores[userID]
            # 遍历此用户在所有类目下的评分，数据结构为字典{cateID:[(itemID,ratings),.....]}
            for user_cateID in user_ratings_dic:
                self.userPos_blank_itemPos[user_pos][user_cateID] = [ ]
                user_cateID_ratings_list = user_ratings_dic[user_cateID]
                itemsINcateID_npa = \
                self.selected_five_category[user_cateID][1]
                itemPos_itemIDs_dic = Drafts_get_itemPos_itemIDs_dic(itemsINcateID_npa)
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
                    try:
                        del itemPos_itemIDs_dic[item_pos]
                    except KeyError:
                        pass
                    print("Update the rating which category:%d, UserID:%d, \
                    UserPos:%d,ItemID:%d, ItemPos:%d !" \
                    % (user_cateID, userID, user_pos, itemID, item_pos))
                itemPos_itemIDs_li = sorted(itemPos_itemIDs_dic.keys())
                self.userPos_blank_itemPos[user_pos][user_cateID] = itemPos_itemIDs_li
        return True

    def update_sparse_matrix_with_check(self):
        '''
        update sparse matrix
        '''
        # 遍历所有的用户, user_pos 为该用户在矩阵的行位置
        #for user_pos in range(len(self.userIDs_pos)):
        user_not_rating_counts = 0    
        user_not_rating_all_cate_counts = 0    
        user_not_rating_in_a_cate = 0
        user_not_rating_2more_in_a_cate = 0
        userID_not_rating_all_cate = [ ]
        for user_pos in range(len(self.userIDs_pos_10ratings)):
            userID = self.userIDs_pos_10ratings[user_pos]
            #user_ratings_dic = self.two_more_ratings_users_dic[userID]
            user_ratings_dic = self.user_has_2more_ratings_in_all_categores[userID]
            if len(user_ratings_dic) is 0:
                user_not_rating_counts += 1
            if len(user_ratings_dic) < 5:
                user_not_rating_all_cate_counts += 1
                userID_not_rating_all_cate.append(userID)
            # 遍历此用户在所有类目下的评分，数据结构为字典{cateID:[(itemID,ratings),.....]}
            for user_cateID in user_ratings_dic:
                user_cateID_ratings_list = user_ratings_dic[user_cateID]
                if len(user_cateID_ratings_list) is 0:
                    user_not_rating_in_a_cate += 1
                if len(user_cateID_ratings_list) < 2:
                    user_not_rating_2more_in_a_cate += 1
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
        print("The number of user which no ratings is %d!" %user_not_rating_counts)
        print("The number of user which no ratings in each category is %d!" %user_not_rating_all_cate_counts)
        print("The number of user which no ratings in a category is %d!" %user_not_rating_in_a_cate)
        print("The number of user which no 2more ratings in a category is %d!" %user_not_rating_2more_in_a_cate)
        return userID_not_rating_all_cate
        return True

    def get_two_ratings_users_in_all_categories(self):
        '''
        To Extract the users who has at least 2 ratings in each selected categories
        '''
        for userID in self.two_more_ratings_users_dic:
            print("Processing User.NO : %d" %userID)
            # get all ratings for a user
            user = self.two_more_ratings_users_dic[userID]
            # The bool flag for weather remain this user, default is True
            remain_flag = True
            for cateID in user:
                # get all ratings in a category for the user
                cate = user[cateID]
                # if the user has 2 more ratings in the catogory, the remainFlag is true
                if len(cate) >= 2:
                    pass
                else:
                    remain_flag = False
                    continue
            
            # check whether save the user
            if remain_flag:
                self.user_has_2more_ratings_in_all_categores[userID] = user
                print("Remain User.NO : %d" %userID)
            else:
                self.user_not_has_2more_ratings_in_all_categores[userID] = user
                print("Delete User.NO : %d" %userID)

        user_10ratings_ID = self.user_has_2more_ratings_in_all_categores.keys()
        self.userIDs_pos_10ratings = sorted(user_10ratings_ID)
        print("Processed all users, WORK DONE")
        
        return True

    def get_two_ratings_users_in_all_categories_with_check(self):
        '''
        To Extract the users who has at least 2 ratings in each selected categories
        '''
        for userID in self.two_more_ratings_users_dic:
            print("Processing User.NO : %d" %userID)
            # get all ratings for a user
            user = self.two_more_ratings_users_dic[userID]
            # The bool flag for weather remain this user, default is True
            remain_flag = True
            RemainFlag = False
            if len(user) == 5:
                RemainFlag = True
            for cateID in user:
                # get all ratings in a category for the user
                cate = user[cateID]
                # if the user has 2 more ratings in the catogory, the remainFlag is true
                if len(cate) >= 2:
                    pass
                else:
                    remain_flag = False
                    continue
            # check whether save the user
            if RemainFlag:
                if remain_flag:
                    self.user_has_2more_ratings_in_all_categores[userID] = user
                    print("Remain User.NO : %d" %userID)
                else:
                    self.user_not_has_2more_ratings_in_all_categores[userID] = user
                    print("Delete User.NO : %d" %userID)
            else:
                self.user_not_has_2more_ratings_in_all_categores[userID] = user
                print("RemainFalg is 0 ----------> Delete User.NO : %d" %userID)
        user_10ratings_ID = self.user_has_2more_ratings_in_all_categores.keys()
        self.userIDs_pos_10ratings = sorted(user_10ratings_ID)
        print("Processed all users, WORK DONE")
        
        return True

    def combine_matrix_userPos_ItemPos(self):
        """
        combine all sparse matrix and userID index and ItemID index
        and return it as a dictionary
        """
        self.WITF_raw_data["userPos"] = self.userIDs_pos
        #self.WITF_raw_data["userPos"] = self.userIDs_pos_10ratings
        self.WITF_raw_data["itemPos"] = self.selected_five_category
        self.WITF_raw_data["matrix"] = self.sparse_matrix_dic
        self.WITF_raw_data["blankCol"] = self.userPos_blank_itemPos
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
#tensor.get_two_ratings_users_in_all_categories()
tensor.init_sparse_matrix_from_txtData()
tensor.update_sparse_matrix_from_txtData()
tensor.combine_matrix_userPos_ItemPos()
#filename1 = "../txtData/UsersHas2MoreRatingsInAllCategoires.txt"
#save_to_txt(tensor.user_has_2more_ratings_in_all_categores,filename1)
#filename2 = "../txtData/Users_NOT_Has2MoreRatingsInAllCategoires.txt"
#save_to_txt(tensor.user_not_has_2more_ratings_in_all_categores,filename2)
filename3 = "/home/Colin/txtData/forWITFs/WITF_raw_data_5_domains.txt"
save_to_txt(tensor.WITF_raw_data,filename3)
print("Saved all data")



# ------------- THE END -----------------------------------------------
