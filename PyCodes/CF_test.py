# Python 3
# Filename: CF_test.py
# Class for Collaborative Filtering Recommender System

import random
import csv
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import scipy

from scipy.sparse import dok_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import identity as SM_identity
from scipy.sparse import random as SM_random
from scipy.sparse import diags as SM_diags
from scipy.sparse import kron as SM_kron
from scipy.sparse import hstack as SM_hstack

from scipy.sparse.linalg import svds as SSL_svds
from scipy.sparse.linalg import inv as SSL_inv
from scipy.sparse.linalg import norm as SSL_ForNorm
#  from tensorly.tenalg import _khatri_rao as TLY_kha_rao

from SaveData import *
from functionDrafts import *

from surprise import BaselineOnly
from surprise import SlopeOne
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import KNNBasic
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


class BenchMarks():
    '''
        class for prepare test and trainset for BenchMarks Recommender System 
    '''
    def __init__(self,dataFile):
       self.data = load_from_txt(dataFile)
       self.obj_value = self.data["objValue"]
       self.iter_times = self.data["IterTimes"]
       self.test_sets = self.data["testSets"]
       self.train_sets = self.data["trainSets"]
       self.userPos = self.data["userPos"]
       self.itemPos = self.data["itemPos"]
       self.U = self.data["U"]
       self.V = self.data["V"]
       self.C = self.data["C"]
       self.Pk = self.data["Pk"]
       print("BenchMarks : Load data from files sucessfully!")
       self.new_testSets_dic = { }

    def ParseTestSets(self):
        """
            Parse Test Sets : Count all tested users
        """
        self.trainSets_itemCount = { }
        counted_items = 0
        test_sets_data = self.test_sets["dataDic"]
        Cate_num = self.test_sets["target_cateID"]
        for cateID in self.train_sets["matrix"].keys():
            item_Count = self.train_sets["matrix"][cateID].shape[1]
            self.trainSets_itemCount[cateID] = (item_Count,counted_items)
            counted_items = counted_items + item_Count
        for user_item_pair in test_sets_data.keys():
            userID = user_item_pair[0]
            itemID = user_item_pair[1] + self.trainSets_itemCount[Cate_num][1]
            rating = test_sets_data[user_item_pair]
            if userID in self.new_testSets_dic:
                self.new_testSets_dic[userID][itemID] = rating
                pass
            else:
                self.new_testSets_dic[userID] = {itemID : rating}
        print("BenchMarks : Parse testSets to new testSets sucessfully!")
        return True

    def ParseTrainSets(self):
        """
            Parse Train Sets : All categories matrix transfer into a whole matrix
        """
        train_mats = self.train_sets["matrix"]
        count = 1
        for cateID in train_mats.keys():
            mat = train_mats[cateID]
            if count is 1:
                Mats = mat
                count += 1
            else:
                Mats = SM_hstack([Mats,mat])
                count += 1
        self.whole_train_matirx = Mats.todok()
        print("BenchMarks : Parse all Cates matrix into a whole train matrix sucessfully!")
        return True

    def run(self):
        """
            main funtion for BenchMarks Class
        """
        print("BenchMarks : Start run()!") 
        self.ParseTrainSets()
        self.ParseTestSets()
        self.UserCF = UserBasedCF(self.whole_train_matirx,self.new_testSets_dic)
        #  self.UserCF.run()
        print("BenchMarks : End run()!") 
        return True

    def making_a_csv(self,dataFile,saveFile):
        """
            making a csv file like MovieLens Datasets (csv file)
        """
        raw_data = load_from_txt(dataFile)
        train_mats = raw_data["matrix"]
        count = 1
        for cateID in train_mats.keys():
            mat = train_mats[cateID]
            if count is 1:
                Mats = mat
                count += 1
            else:
                Mats = SM_hstack([Mats,mat])
                count += 1
        whole_train_matirx = Mats.todok()
        print("Finished get a whole matrix!")
        pbar = tqdm(range(whole_train_matirx.shape[0]))
        LIST = [ ]
        for i in pbar:
            pbar.set_description("Making CSV --> Scan Each user in Matrix:")
            row_i = whole_train_matirx[i]
            row_i_col = row_i.nonzero()[1]
            for j in row_i_col:
                rating = whole_train_matirx[i,j]
                LIST.append((i,j,rating))
        print("Finished get a whole LIST!")
        headers = ['userID','itemID','rating']
        with open(saveFile,'w') as f:
            f_csv = csv.writer(f)
            #  f_csv.writerow(headers)
            f_csv.writerows(LIST)
        print("Writed CSV File!")
        return True 


class UserBasedCF:
    '''
        Class for ItemBased CF
    '''
    def __init__(self,train_matrix,test_sets):
        self.train_matrix = train_matrix
        self.test_sets = test_sets
        print("ItemBasedCF : get trainSet and testSet sucessfully!")

    def find_users_who_rated_this_col(self,col):
        """
            Find all users (except the users in testSets) who have rated this col(item)
        """
        item_col = self.train_matrix[:,col]
        user_who_rated = item_col.nonzero()[0]
        user_who_rated_for_iter = user_who_rated 
        for userID_pos in range(len(user_who_rated_for_iter)):
            userID = user_who_rated[userID_pos]
            if userID in self.train_matrix:
                np.delete(user_who_rated,userID_pos)
        return user_who_rated

    def cal_Similiar(self,user_1,user_2):
        """
            calculate the similiar between 2 users
            Using Cosine Similarity
        """
        U1 = self.train_matrix[user_1].toarray()[0]
        U2 = self.train_matrix[user_2].toarray()[0]
        LEN = len(U1)
        Sum_xy = 0
        Sum_x2 = 0
        Sum_y2 = 0
        for i in range(LEN):
            xy = U1[i]*U2[i]
            Sum_xy += xy
            Sum_x2 += U1[i]**2
            Sum_y2 += U2[i]**2
        similiar = Sum_xy/(math.sqrt(Sum_x2)*math.sqrt(Sum_y2))
        return similiar

    def predict_rating(self,userID,itemID):
        """
            make predict rating for a userID to a itemID
        """
        sim_dic = { }
        sim_li = [ ]
        user_who_rated = self.find_users_who_rated_this_col(itemID)
        for user in user_who_rated:
            similiar = self.cal_Similiar(userID,user)
            if len(sim_dic) > 10:
                sim_li = sorted(sim_dic.items(),key=lambda item:item[1],reverse=True)
                #  sim_li = sim_li.reverse()
                if similiar > sim_li[9][1]:
                   min_sim_user = sim_li.pop()[0]
                   sim_dic.pop(min_sim_user)
                   #  sim_li.append(similiar)
                   sim_dic[user] = similiar
                else:
                    pass
            else:
                #  sim_li.append(similiar)
                sim_dic[user] = similiar
        # 根据筛选出的最相似的10个用户，来进行评分
        sum_rate = 0
        if len(sim_dic) is 0:
            return 0
        for user in sim_dic.keys():
            user_rate = self.train_matrix[user,itemID]
            sum_rate = sum_rate + user_rate
        pre_rating = sum_rate/(len(sim_dic))
        return pre_rating

    def get_Nearset_Neighbors(self,userID):
        """
            Find Nearset Neighbors
            1. 先确认这个用户要预测的商品
            2. 找到其他所有评价过该商品的用户（除了测试集中的用户）
            3. 计算相似度，选取top-K
            4. 预测评分
        """
        #  user_row = self.train_matrix[userID]
        user_test_items = self.test_sets[userID]
        for itemID in user_test_items:
            real_rating = user_test_items[itemID]
            pre_rating = self.predict_rating(userID,itemID)
            pass
        return remain_users

    def get_Nearset_Neighbors_Old(self,userID):
        """
            Find Nearset Neighbors
        """
        user_row = self.train_matrix[userID]
        nonzero_cols = user_row.nonzero()[1]
        count = 0
        for col in nonzero_cols:
            if count is 0:
                user_who_rated = self.find_users_who_rated_this_col(col)
                remain_users = user_who_rated
                #  print(len(remain_users))
                count += 1
            else:
                user_who_rated = self.find_users_who_rated_this_col(col)
                remain_users = np.intersect1d(remain_users,user_who_rated)
                #  print(len(remain_users))
                count += 1
        return remain_users

    def Cal_one_Test_User(self,userID):
        """
            calculate the user's prediction ratings 
            1.计算相似的用户
                --> 找出有相同评分经历的用户
                --> 计算相似度（余弦，etc)
            2.根据相似度进行推荐
        """
        items_ratings = self.test_sets[userID]
        for itemID in items_ratings.keys():
            pre_rating = self.predict_rating(userID,itemID)
            real_rating = items_ratings[itemID]
            self.test_sets[userID][itemID] = (real_rating,pre_rating)
        return True

    def Cal_all_Test_Users(self):
        """
            Calculate all test users one by one
        """
        all_test_user = list(self.test_sets.keys())
        pbar = tqdm(range(len(all_test_user)))
        for i in pbar:
        #  for userID in self.test_sets.keys():
            pbar.set_description("Predict ratings for each test user by User-Based CF:")
            userID = all_test_user[i]
            item_rating_dic = self.test_sets[userID]
            # calculate each user in the test sets
            self.Cal_one_Test_User(userID)
        #  for user
        pbar.close()
        return True

    def run(self):
        """
            ItemBasedCFTest : For Test ItemBased CF 
        """
        print("ItemBasedCF : Start run()!") 
        # step 1 : Calculate predict ratings for all users in testSets
        self.Cal_all_Test_Users()
        filename = "/home/Colin/txtData/Benchmarks/U10I10_noNoises/test_sets_with_preRatings.txt"
        save_to_txt(self.test_sets,filename)
        print("Save TestSets with Predict Ratings Successfully!")
        print("ItemBasedCF : End run()!") 
        return True




class Suprise_Benchmarks:
    '''
        Class for Recommender System Baseline Methods, Based on Scikit Surprise Packages
    '''
    def __init__(self,csv_file):
        self.df = pd.read_csv(csv_file,names=('userID','itemID','rating'))
        self.reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(self.df[['userID', 'itemID', 'rating']], self.reader)
        print("Suprise_Benchmarks : Load data sucessfully!")

    def SVD(self,cv=5):
        """
            Benckmarks SVD algorithms : from surprise import SVD
        """
        data = self.data
        algo = SVD()
        res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        print("Returned the SVD results!")
        return res

    def SVDpp(self,cv=5):
        """
            Benckmarks SVD++ algorithms : from surprise import SVD
        """
        data = self.data
        algo = SVDpp()
        res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        print("Returned the SVDpp results!")
        return res

    def NMF(self,cv=5):
        """
            Benckmarks NMF algorithms : from surprise import SVD
        """
        data = self.data
        algo = NMF()
        res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        print("Returned the NMF results!")
        return res

    def KNNBasic(self,cv=5):
        """
            Benckmarks KNNBasic algorithms : from surprise import SVD
        """
        data = self.data
        algo = KNNBasic()
        res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        print("Returned the KNNBasic results!")
        return res

    def SlopeOne(self,cv=5):
        """
            Benckmarks KNNBasic algorithms : from surprise import SVD
        """
        data = self.data
        algo = SlopeOne()
        res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        print("Returned the SlopeOne results!")
        return res

# -------------------------------------------------------------------
# Main Functions (For Test!!!)
# -------------------------------------------------------------------
data_file = "/home/Colin/txtData/U10I10_Iterated_Data/R5_init1to5_U10I10_mn3_Iter20/No4_iteration.txt"
#  preCom_file = "/home/Colin/GitHubFiles/new_WITF_data/R5_init1to50_mn3_Iter20/.txt"
BM = BenchMarks(data_file)
BM.run()
#  UCF = BM.UserCF
#  BM.ParseTestSets()
#  BM.ParseTrainSets()
FILE = "/home/Colin/GitHubFiles/new_WITF_data/Raw_Datasets/User5_Item5/new_raw_data_for_WITF_py.txt"
SAVE = "/home/Colin/txtData/Benchmarks/All_raw_data_csv_U5I5/all_data_U5I5.csv"
#  allMats = BM.making_a_csv(FILE,SAVE)

#  # ----------------------------------------------------------------------
#  # Suprise Parts
#  # ----------------------------------------------------------------------
#  file_path = SAVE
#  SB = Suprise_Benchmarks(file_path)
#  res = SB.SVD()
#  print(res)
#  res = SB.SVDpp()
#  print(res)
#  res = SB.NMF()
#  print(res)
#  res = SB.KNNBasic()
#  print(res)
#  res = SB.SlopeOne()
#  print(res)
#  reader = Reader(line_format=u'userID,itemID,rating', sep=',')
#  data = Dataset.load_from_file(file_path, reader=reader)
#  df_none = pd.read_csv(file_path,header=None)
#  df = pd.read_csv(file_path,names=('userID','itemID','rating'))
#  reader = Reader(rating_scale=(1, 5))
#  data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
#  print(cross_validate(NormalPredictor(), data, cv=2))

