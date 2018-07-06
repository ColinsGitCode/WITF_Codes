# Python3
# Filename: WITF.py
# Usages : WITF model codes

import numpy as np
import scipy
from scipy.sparse import dok_matrix
from scipy.sparse import coo_matrix

from SaveData import *
from TensorIrr import *
from functionDrafts import *

class WITF:
    ''' Class for the WITF model '''
    
    def __init__(self):
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
        self.raw_data = \
        load_from_txt("../txtData/WITF_raw_data_5_domains.txt")
        self.userPos_li = self.raw_data["userPos"]
        self.itemPos_dic = self.raw_data["itemPos"]
        self.raw_sparMats_dic = self.raw_data["matrix"]
        # the train dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "matrix" : sparMats_dic}
        self.training_sparMats_dic = { "target_cateID" : 0, "ratio" : 0, \
                                       "matrix" : self.raw_sparMats_dic }
        # the test dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "datalist" : [(row,col,ratings),... ]}
        self.test_data_dic = { "target_cateID" : 0, "ratio" : 0, \
                                       "datalist" : [ ] }

    def main_proceduce(self,target_cateID,train_data_ratio):
        """
           The main_proceduce for the WITF model
           Parameter : target_cateID --> choose which category as the target category 
                       train_data_ratio --> set the ratio(%) for the training data and test data
        """
        # 1. Data Preparation
        # 1.1 training data and test data ratios --> function : split_data_byRatios(self,data_ratio)
        self.split_data_byRatios(target_cateID,train_data_ratio)
        print("Finished SPLIT training and test dataset with target_cateID: %d and Ratios: %d Precentages!" \
                 %(target_cateID, 100*train_data_ratio))
        # 1.2 Add virtual data (nosies) into nosies
        #     ---> Function : add_noises(self)

        pass

    def add_noises(self):
        pass

    def split_data_byRatios(self,target_cateID,ratios):
        """
            split the training data and test data by ratio and target_cateID
        """
        self.training_sparMats_dic["target_cateID"] = target_cateID
        self.training_sparMats_dic["ratio"] = ratios
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
            # update a element into test data
            self.test_data_dic["datalist"].append((row,col,rating))
        self.training_sparMats_dic["matrix"][target_cateID] = target_sparMat
        return True

# ================================================================================================
#   Main Fucntions
# ================================================================================================
witf = WITF()
print("Just create a WITF class object witf!")
