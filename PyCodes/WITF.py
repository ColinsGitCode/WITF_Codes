# Python3
# Filename: WITF.py
# Usages : WITF model codes

import random
import numpy as np
import scipy
from scipy.sparse import dok_matrix
from scipy.sparse import coo_matrix

from SaveData import *
#from TensorIrr import *
from functionDrafts import *

class WITF:
    ''' Class for the WITF model '''
    
    def __init__(self,target_cateID=4,ratios=0.8,noiseCount=10):
        """
         load raw data from txt files
         1. sparse matrix : key --> "matrix" , value --> TensorTrr.sparse_matrix_dic(dic)
                ----> sparse_matrix_dic : key --> categoryID, value --> dok_sparse_matrix(userCount,itemCount)
                       ----> dok_sparse_matrix : {[userPos,itemPos] : ratings(np.float16}
         2. userID postion : key --> "userPos", value --> TensorIrr.userIDs_pos_10ratings(list)(sorted)
                ----> [userID1, userID2, ... ] (sorted)
         3. itemID postion : key --> "itemPos", value --> TensorIrr.selected_five_category(dic)
                ----> selected_five_category : key --> categoryID, value --> ("cate_name", nparray([itemID1,..])
        """
        self.target_cateID = target_cateID
        self.ratios = ratios
        self.noiseCount = noiseCount
        self.raw_data = \
        load_from_txt("/home/Colin/txtData/forWITFs/WITF_raw_data_5_domains.txt")
        self.userPos_li = self.raw_data["userPos"]
        self.itemPos_dic = self.raw_data["itemPos"]
        self.raw_sparMats_dic = self.raw_data["matrix"]
        self.blank_cols = self.raw_data["blankCol"]
        # the train dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "matrix" : sparMats_dic}
        self.training_sparMats_dic = { "target_cateID" : 0, "ratio" : 0, \
                                       "matrix" : {} ,"noise" : False}
        # the test dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "datalist" : [(row,col,ratings),... ]}
        #self.test_data_dic = { "target_cateID" : 0, "ratio" : 0, \
        #                               "datalist" : [ ] }
        self.test_data_dic = { "target_cateID" : 0, "ratio" : 0, \
                                       "dataDic" : {} }
        # save the mean, mu_k, sigma for each category
        # DS : self.trainSets_stats_dic = 
        #            { categoryID : { "mean": mean_value, "mu_k" : mu_k, "sigma" : sigma},...}
        self.trainSets_stats_dic = { }

    def main_proceduce(self):
        """
           The main_proceduce for the WITF model
           Parameter : target_cateID --> choose which category as the target category 
                       train_data_ratio --> set the ratio(%) for the training data and test data
        """
        # 1. Data Preparation
        # 1.1 training data and test data ratios --> function : split_data_byRatios(self,data_ratio)
        self.split_data_byRatios()
        print("Finished SPLIT training and test dataset with target_cateID: %d and Ratios: %d Precentages!" \
                 %(self.target_cateID, 100*self.ratios))
        # 1.2 Add virtual data (nosies) into nosies
        #     ---> Function : add_noises(self)

        return True

    def add_noises(self):
        """
            add noises(virtual data) for each user in each category
        """
        for cateID in self.training_sparMats_dic["matrix"]:
            mu_k = self.trainSets_stats_dic[cateID]["mu_k"]
            sigma = self.trainSets_stats_dic[cateID]["sigma"]
            for user_pos in range(len(self.userPos_li)):
                selectd_blanks = random.sample(self.blank_cols[user_pos][cateID], self.noiseCount)
                #selectd_blanks = random.sample(self.userPos_li[user_pos][cateID], self.noiseCount)
                noise_li = np.random.normal(mu_k,sigma,self.noiseCount)
                for index in range(self.noiseCount):
                    blank_col = selectd_blanks[index]
                    noise = noise_li[index]
                    self.training_sparMats_dic["matrix"][cateID][user_pos,blank_col] = noise
        self.training_sparMats_dic["noise"] = True
        return True

    def cal_stats_for_trainSets(self):
        for cateID in self.training_sparMats_dic["matrix"]:
            self.trainSets_stats_dic[cateID] = { }
            sparMat = self.training_sparMats_dic["matrix"][cateID]
            mean_mats = (sparMat.sum())//(len(sparMat.nonzero()[0]))
            if mean_mats == 4:
                sigma = 0.5
            elif mean_mats == 3:
                sigma = 1
            else:
                sigma = 0.5
            self.trainSets_stats_dic[cateID]["mean"] = mean_mats
            self.trainSets_stats_dic[cateID]["mu_k"] = mean_mats
            self.trainSets_stats_dic[cateID]["sigma"] = sigma
        return True

    def split_data_byRatios(self):
        """
            split the training data and test data by ratio and target_cateID
        """
        target_cateID = self.target_cateID
        ratios = self.ratios
        self.test_data_dic["dataDic"] = { }
        self.training_sparMats_dic["target_cateID"] = target_cateID
        self.training_sparMats_dic["ratio"] = ratios
        self.training_sparMats_dic["matrix"] = self.raw_sparMats_dic
        self.test_data_dic["target_cateID"] = target_cateID
        self.test_data_dic["ratio"] = ratios
        target_sparMat = self.raw_sparMats_dic[target_cateID]
        row_NonZero = target_sparMat.nonzero()[0]
        col_NonZero = target_sparMat.nonzero()[1]
        sampled_index = Drafts_samples(len(row_NonZero),1-ratios)
        for index in sampled_index:
            # get each randomly select test data's row and col
            row = row_NonZero[index]
            col = col_NonZero[index]
            rating = target_sparMat[row,col]
            # the randomly select data is deleted(changed to 0 ??)
            target_sparMat[row,col] = 0
            self.test_data_dic["dataDic"][(row,col)] = rating
            # update a element into test data
            # self.test_data_dic["datalist"].append((row,col,rating))
        self.training_sparMats_dic["matrix"][target_cateID] = target_sparMat
        return True

# ================================================================================================
#   Main Fucntions
# ================================================================================================
witf = WITF()
#witf.main_proceduce()
#print("Just create a WITF class object witf!")
