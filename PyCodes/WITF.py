# Python3
# Filename: WITF.py
# Usages : WITF model codes

import random
import numpy as np
import scipy

from scipy.sparse import dok_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import identity as SM_identity
from scipy.sparse import random as SM_random
from scipy.sparse import diags as SM_diags
from scipy.sparse.linalg import svds as SSL_svds

from SaveData import *
#from TensorIrr import *
from functionDrafts import *

class WITF:
    ''' Class for the WITF data pre-computing '''
    
    def __init__(self,target_cateID=4,ratios=0.8,noiseCount=2,add_noise_times=5,R_latent_feature_Num=5):
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
        # ****************************************************************************
        # 基本配置常量
        self.target_cateID = target_cateID
        self.ratios = ratios
        self.noiseCount = noiseCount
        self.add_noise_times = add_noise_times
        self.R_latent_feature_Num = R_latent_feature_Num
        # ****************************************************************************
        # ****************************************************************************
        # 读取由 TensorIrr产生的数据(txt files)
        # self.raw_data.keys() : userPos, itemPos, matrix, ratingsPos
        self.raw_data = \
        load_from_txt("/home/Colin/txtData/forWITFs/WITF_raw_data_5_domains.txt")
        # ****************************************************************************
        # ****************************************************************************
        # 任意分类有两个以上分类的用户ID的排序列表
        # self.userPos_li = [userID1, userID2, ..., userID307704 ](sorted)(20338)
        self.userPos_li = self.raw_data["userPos"]
        # ****************************************************************************
        # ****************************************************************************
        # 选择的五个分类的所有的itemID, 保存在nparray中
        # DS : self.itemPos_dic = 
        #            { cateID : ("cate_name" , ndarray[itemID,itemID,...](sorted)), ... }
        self.itemPos_dic = self.raw_data["itemPos"]
        # ****************************************************************************
        # ****************************************************************************
        self.raw_sparMats_dic = self.raw_data["matrix"]
        # ****************************************************************************
        # ****************************************************************************
        # 保存每个用户的评分的(ratings_postions)位置
        # DS : self.userPos_ratings_itemPos = 
        #              {userPos : {cateID : (bool, [RatingItemsPos1, RatingItemsPos2, ...](sorted) ), ...}, ... }
        self.userPos_ratings_itemPos = self.raw_data["ratingPos"]
        # ****************************************************************************
        # ****************************************************************************
        # the train dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "matrix" : sparMats_dic}
        self.training_sparMats_dic = { "target_cateID" : 0, "ratio" : 0, \
                                       "matrix" : {} ,"noise" : False}
        # ****************************************************************************
        # ****************************************************************************
        # the test dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "datalist" : [(row,col,ratings),... ]}
        #self.test_data_dic = { "target_cateID" : 0, "ratio" : 0, \
        #                               "datalist" : [ ] }
        self.test_data_dic = { "target_cateID" : 0, "ratio" : 0, \
                                       "dataDic" : {} }
        # ****************************************************************************
        # ****************************************************************************
        # save the mean, mu_k, sigma for each category
        # DS : self.trainSets_stats_dic = 
        #            { categoryID : { "mean": mean_value, "mu_k" : mu_k, "sigma" : sigma}
        #                             "allItemPos" : { itemPos: True, ... },...}
        self.trainSets_stats_dic = { }
        # ****************************************************************************
        # ****************************************************************************
        # self.user_noisePos = { user_pos : {cateID : [[ ],[ ], ... ], ... }, ... }
        self.user_noisePos = { }
        # ****************************************************************************
        # ****************************************************************************
        # Save Constrant {Pk} for each category
        # self.P_k_dic = { cateID : matrix P_k , ... }
        self.P_k_dic = { }
        # ****************************************************************************
        # ****************************************************************************
        # Save Weight matrixs: W_kij
        # self.P_k_dic = { cateID : weights over ratings matrix , ... }
        self.ratings_weights_matrixs_dic = { }
        # ****************************************************************************
        # 无用的变量
        self.all_blank_pos_dic = { }
        self.blank_cols = { }

    def main_proceduce(self):
        """
           The main_proceduce for the WITF model
           Parameter : target_cateID --> choose which category as the target category 
                       train_data_ratio --> set the ratio(%) for the training data and test data
        """
        self.split_data_byRatios()
        print("Finished SPLIT training and test dataset with target_cateID: %d and Ratios: %d Precentages!" \
                 %(self.target_cateID, 100*self.ratios))
        self.cal_stats_for_trainSets()
        print("Finished calculate the stats for trianing datasets!")
        self.cal_users_noisePos()
        self.randomly_init_U_C_V()
        print("Finshed randomly_init_U_C_V() Functions")
        self.find_Pk()
        print("Finshed find_Pk() Functions")
        self.init_ratings_weights_matrix()
        print("Finished init_ratings_weights_matrix() Functions")
        self.set_observation_weights()
        print("Finished set_observation_weights() Functions")
        self.save_PreComputed_data()
        print("Finshed save_PreComputed_data() Functions")
        return True

    def save_PreComputed_data(self):
        """
            save the pre-computed data as txt files for WITF_Iterations class
        """
        saveData = { }
        saveData["trainSets"] = self.training_sparMats_dic
        saveData["ratingWeights"] = self.ratings_weights_matrixs_dic
        saveData["testSets"] = self.test_data_dic
        saveData["targetCateID"] = self.target_cateID
        saveData["ratios"] = self.ratios
        saveData["noiseCount"] = self.noiseCount
        saveData["noiseTimes"] = self.add_noise_times
        saveData["R_sizes"] = self.R_latent_feature_Num
        saveData["userPos"] = self.userPos_li
        saveData["itemPos"] = self.itemPos_dic
        saveData["trainSetStats"] = self.trainSets_stats_dic
        saveData["noisePos"] = self.user_noisePos
        saveData["P_k"] = self.P_k_dic
        saveData["U"] = self.U_Mats
        saveData["V"] = self.V_Mats
        saveData["C"] = self.C_Mats
        filename = "/home/Colin/txtData/forWITFs/WITF_Pre_Computed_Data.txt"
        save_to_txt(saveData,filename)
        return True

    def find_Pk(self):
        """
            Calculate the constrant Pk for each category by SVD
        """
        catelist = self.training_sparMats_dic["matrix"].keys()
        cate_list = [ ]
        for key in catelist:
            cate_list.append(key)
        for cate_index in range(len(catelist)):
        #  for cateID in self.training_sparMats_dic["matrix"]:
            cateID = cate_list[cate_index]
            X_k = self.training_sparMats_dic["matrix"][cateID]
            C_k_row = self.C_Mats.getrow(cate_index).toarray()[0]
            Sigma_k = SM_diags(C_k_row)
            SVD_mats = X_k.T
            SVD_mats = SVD_mats.dot(self.U_Mats)
            SVD_mats = SVD_mats.dot(Sigma_k)
            SVD_mats = SVD_mats.dot(self.V_Mats.T)
            # SVD R select questions??
            #  A_r, sigma_r, B_r_t = SSL_svds(SVD_mats)
            A_r, sigma_r, B_r_t = SSL_svds(SVD_mats,k=self.R_latent_feature_Num-1)
            P_k = coo_matrix(A_r.dot(B_r_t))
            self.P_k_dic[cateID] = P_k
        return True


    def randomly_init_U_C_V(self):
        """
            randomly init the latent feature matrices: U, C, V
        """
        U_N_userNum = len(self.userPos_li)
        R = self.R_latent_feature_Num
        C_K_cateNum = 5
        # init V as scipy sparse identity matrix
        self.V_Mats = SM_identity(R)
        # Randomly init U, C with density = 1?
        self.U_Mats = SM_random(U_N_userNum,R,density=1,format='dok')
        self.C_Mats = SM_random(C_K_cateNum,R,density=1,format='dok')
        return True

    def cal_users_noisePos(self):
        """
            The function to calculate the noises postions for each user in each category
        """
        for user_pos in self.userPos_ratings_itemPos:
            self.user_noisePos[user_pos] = { }
            ratingPos_dic = self.userPos_ratings_itemPos[user_pos]
            for cateID in ratingPos_dic:
                self.user_noisePos[user_pos][cateID] = [ ]
                rating_cateID = ratingPos_dic[cateID]
                if rating_cateID[0] is False:
                    for time in range(self.add_noise_times):
                        selectedPos = random.sample(self.trainSets_stats_dic[cateID]["allItemPos"].keys(),self.noiseCount)
                        self.user_noisePos[user_pos][cateID].append(selectedPos)
                else:
                    userPos_rating_cateID = self.userPos_ratings_itemPos[user_pos][cateID][1]
                    all_itemPos_cateID_dic = self.trainSets_stats_dic[cateID]["allItemPos"]
                    for itemPos in userPos_rating_cateID:
                        try:
                            del all_itemPos_cateID_dic[itemPos]
                        except KeyError:
                            pass
                    blankPos = sorted(all_itemPos_cateID_dic.keys())
                    for time in range(self.add_noise_times):
                        selectedPos = random.sample(blankPos,self.noiseCount)
                        self.user_noisePos[user_pos][cateID].append(selectedPos)
                print("Finished calculate %d group users noises postions for userPos:%d in cateID:%d!" %(self.add_noise_times,user_pos,cateID)) 
            print("Finished userPos:%d <--------------------> " %user_pos)
        print("Finshed cal_users_noisePos() functions!!!")
        return True

    def add_noises(self):
        """
            add noises(virtual data) for each user in each category
        """
        for cateID in self.training_sparMats_dic["matrix"]:
            mu_k = self.trainSets_stats_dic[cateID]["mu_k"]
            sigma = self.trainSets_stats_dic[cateID]["sigma"]
            for user_pos in range(len(self.userPos_li)):
                #  try:
                    #  selectd_blanks = random.sample(self.blank_cols[user_pos][cateID], self.noiseCount)
                #  except KeyError:
                    #  selectd_blanks = random.sample(self.all_blank_pos_dic[cateID],self.noiseCount)
                #selectd_blanks = random.sample(self.userPos_li[user_pos][cateID], self.noiseCount)
                selectedPos = self.user_noisePos[user_pos][cateID][0]
                noise_li = np.random.normal(mu_k,sigma,self.noiseCount)
                for index in range(self.noiseCount):
                    blank_col = selectedPos[index]
                    #  blank_col = selectd_blanks[index]
                    noise = noise_li[index]
                    self.training_sparMats_dic["matrix"][cateID][user_pos,blank_col] = noise
        self.training_sparMats_dic["noise"] = True
        return True

    def add_noises_verion_1(self):
        """
            add noises(virtual data) for each user in each category
        """
        for cateID in self.training_sparMats_dic["matrix"]:
            mu_k = self.trainSets_stats_dic[cateID]["mu_k"]
            sigma = self.trainSets_stats_dic[cateID]["sigma"]
            for user_pos in range(len(self.userPos_li)):
                try:
                    selectd_blanks = random.sample(self.blank_cols[user_pos][cateID], self.noiseCount)
                except KeyError:
                    selectd_blanks = random.sample(self.all_blank_pos_dic[cateID],self.noiseCount)
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
            #  self.all_blank_pos_dic[cateID] = [ ]
            self.trainSets_stats_dic[cateID] = { }
            #  self.cateID_itemPos_dic[cateID] = { }
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
            itemCount = len(self.itemPos_dic[cateID][1])
            # 用于存储所有itemPos的字典
            itemPos_dic = { }
            for itemPos in range(itemCount):
                #  itemPos_li.append(itemPos)
                itemPos_dic[itemPos] = True
            #  self.cateID_itemPos_dic[cateID] = itemPos_dic
            self.trainSets_stats_dic[cateID]["allItemPos"] = itemPos_dic
            #  self.all_blank_pos_dic[cateID] = itemPos_li
        return True

    def bak_cal_stats_for_trainSets(self):
        for cateID in self.training_sparMats_dic["matrix"]:
            self.all_blank_pos_dic[cateID] = [ ]
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
            itemCount = len(self.itemPos_dic[cateID][1])
            itemPos_li = [ ]
            for itemPos in range(itemCount):
                itemPos_li.append(itemPos)
            self.all_blank_pos_dic[cateID] = itemPos_li
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
        self.training_sparMats_dic["matrix"] = self.raw_sparMats_dic.copy()
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

    def init_ratings_weights_matrix(self):
        '''
        To init the ratings weights matrices 
        '''
        users_conuts = len(self.userPos_li)  # 以前的用户选择标准
        for cate in self.itemPos_dic:
            items_counts = len(self.itemPos_dic[cate][1])
            sm = dok_matrix((users_conuts,items_counts),
                    dtype=np.float32)
            self.ratings_weights_matrixs_dic[cate] = sm
            print("ratings_weights_matrix for category: %d, size is%d and %d!"
                    %(cate,users_conuts,items_counts))
        return True

    def set_observation_weights(self):
        for cateID in self.ratings_weights_matrixs_dic:
            tarining_mats = self.training_sparMats_dic["matrix"][cateID]
            row_NonZero = tarining_mats.nonzero()[0]
            col_NonZero = tarining_mats.nonzero()[1]
            for index in range(len(row_NonZero)):
                row = row_NonZero[index]
                col = col_NonZero[index]
                self.ratings_weights_matrixs_dic[cateID][row,col] = 1
            print("Finished to set observation weight for cateID:%d!!" %cateID)
        return True

# ================================================================================================
#   Global Fucntions
# ================================================================================================

# ================================================================================================
#   Main Fucntions
# ================================================================================================
witf = WITF()
witf.main_proceduce()
#  cate4 = witf.training_sparMats_dic["matrix"][4]
#  cate4_t = cate4.T
#  CATE = cate4_t.dot(cate4)
#  U, SIGMA, V_t = SSL_svds(CATE,k=5)
#print("Just create a WITF class object witf!")
