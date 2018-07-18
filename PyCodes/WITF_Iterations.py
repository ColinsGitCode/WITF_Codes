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
from scipy.sparse import kron as SM_kron

from scipy.sparse.linalg import svds as SSL_svds
from scipy.sparse.linalg import inv as SSL_inv
#  from tensorly.tenalg import _khatri_rao as TLY_kha_rao

from SaveData import *
#from TensorIrr import *
from functionDrafts import *
#from WITF import *

class WITF_Iterations:
    ''' Class for doing the WITF Iterations! '''
    def __init__(self,filename):
        saveData = load_from_txt(filename)
        # ****************************************************************************
        # 基本配置常量
        self.target_cateID = saveData["targetCateID"]  
        self.ratios = saveData["ratios"]  
        self.noiseCount = saveData["noiseCount"]  
        self.add_noise_times = saveData["noiseTimes"]  
        self.R_latent_feature_Num = saveData["R_sizes"] 
        # ****************************************************************************
        # ****************************************************************************
        # the train dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "matrix" : sparMats_dic}
        self.training_sparMats_dic = saveData["trainSets"] 
        # ****************************************************************************
        # ****************************************************************************
        # Save Weight matrixs: W_kij
        # self.P_k_dic = { cateID : weights over ratings matrix , ... }
        self.ratings_weights_matrixs_dic = saveData["ratingWeights"] 
        # ****************************************************************************
        # ****************************************************************************
        # the test dataset = 
        # { "target_cateID" : target_cateID, "ratio" : Ratios, "datalist" : [(row,col,ratings),... ]}
        #self.test_data_dic = { "target_cateID" : 0, "ratio" : 0, \
        #                               "datalist" : [ ] }
        self.test_data_dic = saveData["testSets"]  
        # ****************************************************************************
        # ****************************************************************************
        # 任意分类有两个以上分类的用户ID的排序列表
        # self.userPos_li = [userID1, userID2, ..., userID307704 ](sorted)(20338)
        self.userPos_li = saveData["userPos"]  
        # ****************************************************************************
        # ****************************************************************************
        # 选择的五个分类的所有的itemID, 保存在nparray中
        # DS : self.itemPos_dic = 
        #            { cateID : ("cate_name" , ndarray[itemID,itemID,...](sorted)), ... }
        self.itemPos_dic = saveData["itemPos"]  
        # ****************************************************************************
        # ****************************************************************************
        # save the mean, mu_k, sigma for each category
        # DS : self.trainSets_stats_dic = 
        #            { categoryID : { "mean": mean_value, "mu_k" : mu_k, "sigma" : sigma}
        #                             "allItemPos" : { itemPos: True, ... },...}
        self.trainSets_stats_dic = saveData["trainSetStats"]  
        # ****************************************************************************
        # ****************************************************************************
        # self.user_noisePos = { user_pos : {cateID : [[ ],[ ], ... ], ... }, ... }
        self.user_noisePos = saveData["noisePos"]  
        # ****************************************************************************
        # ****************************************************************************
        # Save Constrant {Pk} for each category
        # self.P_k_dic = { cateID : matrix P_k , ... }
        self.P_k_dic = saveData["P_k"] 
        # ****************************************************************************
        # ****************************************************************************
        # init latent feature matrix
        self.U_Mats = saveData["U"] 
        self.V_Mats = saveData["V"] 
        self.C_Mats = saveData["C"] 
        self.Wkij_dic = saveData["omiga_ki"] 
        # ****************************************************************************
        # ****************************************************************************
        # self.cate_list = [4,17,24,29,40]
        self.cate_list = [ ]
        for key in self.itemPos_dic.keys():
            self.cate_list.append(key)
        # ****************************************************************************
        # ****************************************************************************
        # Svae tensor Y mode-n npa
        #  self.Y_n_dic = { }
        FILENAME = "/home/Colin/txtData/forWITFs/Y_n_dic.txt"
        self.Y_n_dic = load_from_txt(FILENAME)



    def main_proceduce(self):
        """
            main_proceduce for WITF_Iterations class
            1. add_noises(): add noise for each user
            2. assign_weights() assgin weights by a specific weights configurations
                2.1 How to decide the weights, now set as all 1
            3. Iterations (How to decided whether is convergences??????)
               3.1 Sub-Iterations m (1~5): m=1
               3.2 Sub-Iterations n (1~5): n=1
        """
        self.add_noises()
        print("Finished add_noises()!")
        return True

    def assign_weights(self):
        """
            function to assgin weights to irregular tensor
        """
        # now set as 1,1,1,1,1
        pass
        return True

    def sub_iterations(self,user_Count,m=1,n=1):
        """
            Functions for do the sub_iterations
        """
        # The number of categories
        num_K = len(self.cate_list)
        num_N = len(self.userPos_li)
        # sun iteration 1 : update U_i for each user
        for m_iter_time in range(m):
            for user_pos in range(user_Count):
            #  for user_pos in range(1000):
            #  for user_pos in range(len(self.userPos_li)):
                # update row(user_pos) of U by formula 23
                U_i_npa = self.Formula_Ui_23(num_K,user_pos)
                self.U_Mats[user_pos,:] = U_i_npa
                print("Update U_%d !" %user_pos)
            print("Finished Update U_i !")
            for cate_index in range(len(self.cate_list)):
                # update row(cate_index) of C by formula 24
                num_N = user_Count
                #  num_N = 1000
                C_k_npa = self.Formula_Ck_24(num_N,cate_index)
                self.U_Mats[user_pos,:] = C_k_npa
                print("Update C_%d !" %cate_index)
            # Update the whole V by formula 25
            # self.Formula_V_25
            pass
        # sub iteration 2: Update Pk for each domian k using formula 22
        for n_iter_time in range(n):
            # 在此更新Pk,可以使用其他WITF文件里的方法
            # self.Formula_Pk_22
            pass
        print("Finished Sub-Iteration(Old Version) with UserCount:%d !" %user_Count)
        return True

    def Formula_Ui_23(self,num_K=5,user_pos=0):
        """
            function for Formula 23 to update Ui for user i
        """
        # the number for selected categorties
        #  num_K = len(self.cate_list)
        # self.Formula_Ui_23_part1(num_K,user_pos)
        npa_Y_1_i = self.Y_n_dic["Y_1"][user_pos]
        length = npa_Y_1_i.shape[0]
        mat_Y_1_i = dok_matrix((1,length))
        mat_Y_1_i[0,:] = npa_Y_1_i
        C_khaRao_V = self.cal_Khatri_Rao(self.C_Mats,self.V_Mats)
        res_mat = mat_Y_1_i.dot(C_khaRao_V)
        res_mat = res_mat.dot(self.Formula_Ui_23_part2(num_K,user_pos))
        res_npa = res_mat.toarray()[0]
        return res_npa

    def Formula_Ui_23_part1(self,num_K=5,user_pos=0):
        """
            function for calculate Formula 23 part_1
        """
        result_mats = 0
        for k in range(num_K):
            cateID = self.cate_list[k]
            C_k = self.C_Mats.getrow(k).toarray()[0]
            sigma_k = SM_diags(C_k)
            P_k = self.P_k_dic[cateID]
            #  P_k = self.P_k_dic[self.cate_list[k]]
            W_kij = self.ratings_weights_matrixs_dic[cateID]
            # user_pos means i
            W_ki = W_kij.getrow(user_pos).toarray()[0]
            omiga_ki = SM_diags(W_ki)
            size = omiga_ki.shape[0]
            I_omiga_ki = SM_identity(size)
            omiga_ki = omiga_ki - I_omiga_ki
            mats = P_k.dot(self.V_Mats)
            mats = sigma_k.dot(mats.T)
            mats = mats.dot(omiga_ki)
            mats = mats.dot(P_k)
            mats = mats.dot(self.V_Mats)
            mats = mats.dot(sigma_k)
            result_mats = result_mats + mats
            print("In Formula 23 part1 : Done Cate_index:%d !" %k)
        return result_mats.tocsc()

    def Formula_Ui_23_part1_PreCom(self,num_K=5,user_pos=0):
        """
            function for calculate Formula 23 part_1
        """
        result_mats = 0
        for k in range(num_K):
            cateID = self.cate_list[k]
            C_k = self.C_Mats.getrow(k).toarray()[0]
            sigma_k = SM_diags(C_k)
            P_k = self.P_k_dic[cateID]
            #  P_k = self.P_k_dic[self.cate_list[k]]
            #  W_kij = self.ratings_weights_matrixs_dic[cateID]
            # user_pos means i
            #  W_ki = W_kij.getrow(user_pos).toarray()[0]
            #  omiga_ki = SM_diags(W_ki)
            #  size = omiga_ki.shape[0]
            #  I_omiga_ki = SM_identity(size)
            #  omiga_ki = omiga_ki - I_omiga_ki
            omiga_ki = self.Wkij_dic[cateID][user_pos]
            mats = P_k.dot(self.V_Mats)
            mats = sigma_k.dot(mats.T)
            mats = mats.dot(omiga_ki)
            mats = mats.dot(P_k)
            mats = mats.dot(self.V_Mats)
            mats = mats.dot(sigma_k)
            result_mats = result_mats + mats
            print("In Formula 23 part1 : Done Cate_index:%d !" %k)
        return result_mats.tocsc()


    def Formula_Ui_23_part2(self,num_K=5,user_pos=0):
    #  def Formula_Ui_23_part2(self,num_K,user_pos):
        """
            function to calculate the part2 of Formula 23
        """
        C_Mats = self.C_Mats#.tocsc()
        V_Mats = self.V_Mats#.tocsc()
        res_mat1 = ((C_Mats.T).dot(C_Mats))#.tocsc()  
        res_mat2 = ((V_Mats.T).dot(V_Mats))#.tocsc()  
        res_mats = (res_mat1.multiply(res_mat2)).tocsc()
        I_RR = SM_identity(self.R_latent_feature_Num,format='csc')
        res_mats = res_mats + I_RR + self.Formula_Ui_23_part1(num_K,user_pos)
        res_mats = SSL_inv(res_mats)
        return res_mats

    def cal_Khatri_Rao(self,A,B):
        """
            function to calculate the Khatri_Rao Product of 2 matrix (A,B)
        """
        col_num = A.shape[1]
        row_num = (A.shape[0])*(B.shape[0])
        product = dok_matrix((row_num,col_num))
        for col_index in range(col_num):
            col_mat = SM_kron(A.getcol(col_index),B.getcol(col_index))
            product[:,col_index] = col_mat
        return product

    def Formula_Ck_24(self,num_N=10,cate_list=0):
        """
            function for Formula 24 to update Ck for category k
        """
        npa_Y_3_k = self.Y_n_dic["Y_3"][cate_list]
        length = npa_Y_3_k.shape[0]
        mat_Y_3_k = dok_matrix((1,length))
        mat_Y_3_k[0,:] = npa_Y_3_k
        V_khaRao_U = self.cal_Khatri_Rao(self.V_Mats,self.U_Mats)
        res_mat = mat_Y_3_k.dot(V_khaRao_U)
        res_mat = res_mat.dot(self.Formula_Ck_24_part2(num_N,cate_list))
        res_npa = res_mat.toarray()[0]
        return res_npa

    def Formula_Ck_24_part2(self,num_N=10,cate_index=0):
        """
            function to calculate the Formula 24 part2
        """
        V_Mats = self.V_Mats
        U_Mats = self.U_Mats
        res_mat1 = ((U_Mats.T).dot(U_Mats))#.tocsc()  
        res_mat2 = ((V_Mats.T).dot(V_Mats))#.tocsc()  
        res_mats = (res_mat1.multiply(res_mat2)).tocsc()
        I_RR = SM_identity(self.R_latent_feature_Num,format='csc')
        res_mats = res_mats + I_RR + self.Formula_Ck_24_part1(num_N,cate_index)
        res_mats = SSL_inv(res_mats)
        return res_mats

    def Formula_Ck_24_part1(self,num_N=10,cate_index=0):
        """
            function to calculate the Formula 24 part1
        """
        result_mats = 0
        for i in range(num_N):
            cateID = self.cate_list[cate_index]
            U_i = self.U_Mats.getrow(i).toarray()[0]
            sigma_i = SM_diags(U_i)
            P_k = self.P_k_dic[cateID]
            W_kij = self.ratings_weights_matrixs_dic[cateID]
            W_ki = W_kij.getrow(i).toarray()[0]
            omiga_ki = SM_diags(W_ki)
            size = omiga_ki.shape[0]
            I_omiga_ki = SM_identity(size)
            omiga_ki = omiga_ki - I_omiga_ki
            mats = P_k.dot(self.V_Mats)
            mats = sigma_i.dot(mats.T)
            mats = mats.dot(omiga_ki)
            mats = mats.dot(P_k)
            mats = mats.dot(self.V_Mats)
            mats = mats.dot(sigma_i)
            result_mats = result_mats + mats
            print("In Formula 24 part1 : Done UserPos:%d !" %i)
        return result_mats.tocsc()

    def Formula_Ck_24_part1_PreCom(self,num_N=10,cate_index=0):
        """
            function to calculate the Formula 24 part1
        """
        result_mats = 0
        for i in range(num_N):
            cateID = self.cate_list[cate_index]
            U_i = self.U_Mats.getrow(i).toarray()[0]
            sigma_i = SM_diags(U_i)
            P_k = self.P_k_dic[cateID]
            #  W_kij = self.ratings_weights_matrixs_dic[cateID]
            #  W_ki = W_kij.getrow(i).toarray()[0]
            #  omiga_ki = SM_diags(W_ki)
            #  size = omiga_ki.shape[0]
            #  I_omiga_ki = SM_identity(size)
            #  omiga_ki = omiga_ki - I_omiga_ki
            omiga_ki = self.Wkij_dic[cateID][i]
            mats = P_k.dot(self.V_Mats)
            mats = sigma_i.dot(mats.T)
            mats = mats.dot(omiga_ki)
            mats = mats.dot(P_k)
            mats = mats.dot(self.V_Mats)
            mats = mats.dot(sigma_i)
            result_mats = result_mats + mats
            print("In Formula 24 part1 : Done UserPos:%d !" %i)
        return result_mats.tocsc()

    def get_Y_n(self):
        """
            function to get tensor Y mode-n unfolding
        """
        User_num = len(self.userPos_li)
        Cate_num = len(self.cate_list)
        V_num = self.R_latent_feature_Num
        R_num = self.R_latent_feature_Num
        Y = np.random.rand(User_num,V_num,Cate_num)
        for u in range(User_num):
            U_u = self.U_Mats.getrow(u)#.toarray[0]
            for v in range(V_num):
                V_v = self.V_Mats.getrow(v)#.toarray[0]
                for c in range(Cate_num):
                    C_c = self.C_Mats.getrow(v)#.toarray[0]
                    entry = U_u.multiply(V_v)
                    entry = entry.multiply(C_c).sum()
                    Y[u][v][c] = entry
                    print("get_Y_n: Done userPos:%d,VPos:%d,CPos:%d!" %(u,v,c))
        self.Y_n_dic["Y_1"] = np.reshape(np.moveaxis(Y,0,0),(Y.shape[0], -1),order='F')
        self.Y_n_dic["Y_2"] = np.reshape(np.moveaxis(Y,1,0),(Y.shape[1], -1),order='F')
        self.Y_n_dic["Y_3"] = np.reshape(np.moveaxis(Y,2,0),(Y.shape[2], -1),order='F')
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
        

    def sub_iterations_UVC(self,user_Count,m=1):
        """
            Functions for do the sub_iterations
        """
        # The number of categories
        num_K = len(self.cate_list)
        num_N = len(self.userPos_li)
        # Pre-Computing Parts
        C_khaRao_V = self.cal_Khatri_Rao(self.C_Mats,self.V_Mats)
        V_khaRao_U = self.cal_Khatri_Rao(self.V_Mats,self.U_Mats)
        CtCVtV_I = self.cal_AtA_BtB_I(self.C_Mats,self.V_Mats)
        VtVUtU_I = self.cal_AtA_BtB_I(self.V_Mats,self.U_Mats)
        # sun iteration 1 : update U_i for each user
        for m_iter_time in range(m):
            for user_pos in range(user_Count):
            #  for user_pos in range(1000):
            #  for user_pos in range(len(self.userPos_li)):
                # update row(user_pos) of U by formula 23
                U_i_npa = self.Formula_Ui_23_PreCom(num_K,user_pos,C_khaRao_V,CtCVtV_I)
                self.U_Mats[user_pos,:] = U_i_npa
                print("Update U_%d !" %user_pos)
            print("Finished Update U_i !")
            for cate_index in range(len(self.cate_list)):
                # update row(cate_index) of C by formula 24
                num_N = user_Count
                #  num_N = 1000
                C_k_npa = self.Formula_Ck_24_PreCom(num_N,cate_index,V_khaRao_U,VtVUtU_I)
                self.C_Mats[cate_index,:] = C_k_npa
                print("Update C_%d !" %cate_index)
            # Update the whole V by formula 25
            # self.Formula_V_25
        # sub iteration 2: Update Pk for each domian k using formula 22
        for n_iter_time in range(1):
            # 在此更新Pk,可以使用其他WITF文件里的方法
            # self.Formula_Pk_22
            pass
        print("Finished Sub-Iteration(New Version) with UserCount:%d !" %user_Count)
        return True

    def Formula_Ck_24_PreCom(self,num_N,cate_list,V_khaRao_U,VtVUtU_I):
        """
            function for Formula 24 to update Ck for category k
        """
        npa_Y_3_k = self.Y_n_dic["Y_3"][cate_list]
        length = npa_Y_3_k.shape[0]
        mat_Y_3_k = dok_matrix((1,length))
        mat_Y_3_k[0,:] = npa_Y_3_k
        #  V_khaRao_U = self.cal_Khatri_Rao(self.V_Mats,self.U_Mats)
        res_mat = mat_Y_3_k.dot(V_khaRao_U)
        res_mat = res_mat.dot(self.Formula_Ck_24_part2_PreCom(num_N,cate_list,VtVUtU_I))
        res_npa = res_mat.toarray()[0]
        return res_npa

    def Formula_Ck_24_part2_PreCom(self,num_N,cate_index,VtVUtU_I):
        """
            function to calculate the Formula 24 part2
        """
        #  V_Mats = self.V_Mats
        #  U_Mats = self.U_Mats
        #  res_mat1 = ((U_Mats.T).dot(U_Mats))#.tocsc()  
        #  res_mat2 = ((V_Mats.T).dot(V_Mats))#.tocsc()  
        #  res_mats = (res_mat1.multiply(res_mat2)).tocsc()
        #  I_RR = SM_identity(self.R_latent_feature_Num,format='csc')
        #  res_mats = res_mats + I_RR + self.Formula_Ck_24_part1(num_N,cate_index)
        res_mats = VtVUtU_I + self.Formula_Ck_24_part1_PreCom(num_N,cate_index)
        #  res_mats = VtVUtU_I + self.Formula_Ck_24_part1(num_N,cate_index)
        res_mats = SSL_inv(res_mats)
        return res_mats

    def Formula_Ui_23_PreCom(self,num_K,user_pos,C_khaRao_V,CtCVtV_I):
        """
            function for Formula 23 to update Ui for user i
        """
        npa_Y_1_i = self.Y_n_dic["Y_1"][user_pos]
        length = npa_Y_1_i.shape[0]
        mat_Y_1_i = dok_matrix((1,length))
        mat_Y_1_i[0,:] = npa_Y_1_i
        #C_khaRao_V = self.cal_Khatri_Rao(self.C_Mats,self.V_Mats)
        res_mat = mat_Y_1_i.dot(C_khaRao_V)
        res_mat = res_mat.dot(self.Formula_Ui_23_part2_PreCom(num_K,user_pos,CtCVtV_I))
        res_npa = res_mat.toarray()[0]
        return res_npa

    def Formula_Ui_23_part2_PreCom(self,num_K,user_pos,CtCVtV_I):
    #  def Formula_Ui_23_part2(self,num_K,user_pos):
        """
            function to calculate the part2 of Formula 23
        """
        #  C_Mats = self.C_Mats#.tocsc()
        #  V_Mats = self.V_Mats#.tocsc()
        #  res_mat1 = ((C_Mats.T).dot(C_Mats))#.tocsc()  
        #  res_mat2 = ((V_Mats.T).dot(V_Mats))#.tocsc()  
        #  res_mats = (res_mat1.multiply(res_mat2)).tocsc()
        #  I_RR = SM_identity(self.R_latent_feature_Num,format='csc')
        #  res_mats = res_mats + I_RR + self.Formula_Ui_23_part1(num_K,user_pos)
        res_mats = CtCVtV_I + self.Formula_Ui_23_part1_PreCom(num_K,user_pos)
        #  res_mats = CtCVtV_I + self.Formula_Ui_23_part1(num_K,user_pos)
        res_mats = SSL_inv(res_mats)
        return res_mats

    def cal_AtA_BtB_I(self,A,B):
        C_Mats = A
        V_Mats = B
        res_mat1 = ((C_Mats.T).dot(C_Mats))#.tocsc()  
        res_mat2 = ((V_Mats.T).dot(V_Mats))#.tocsc()  
        res_mats = (res_mat1.multiply(res_mat2)).tocsc()
        I_RR = SM_identity(self.R_latent_feature_Num,format='csc')
        res_mats = res_mats + I_RR 
        return res_mats
# ------------------------------------------------------------------------------------------------------
# main functions
# ------------------------------------------------------------------------------------------------------
txtfile = "/home/Colin/txtData/forWITFs/WITF_Pre_Computed_Data.txt"
IWITF = WITF_Iterations(txtfile)
print("Created the instant of WITF_Iterations class which named IWITF!")
IWITF.main_proceduce()
#  IWITF.sub_iterations(100)
IWITF.sub_iterations_UVC(100)

