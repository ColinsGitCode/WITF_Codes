# Python3
# Filename: WITF.py
# Usages : WITF model codes

from tqdm import tqdm
import datetime
import random
import math
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
from scipy.sparse.linalg import norm as SSL_ForNorm
#  from tensorly.tenalg import _khatri_rao as TLY_kha_rao

from SaveData import *
#from TensorIrr import *
from functionDrafts import *
#from WITF import *

class WITF_Iterations:
    ''' Class for doing the WITF Iterations! '''
    def __init__(self,filename,SaveDir,m=3,n=3):
        saveData = load_from_txt(filename)
        print("------------------------------------------------------------")
        print("Loaded PreComputed Data File is %s! " %filename)
        self.SaveDir = SaveDir
        print("------------------------------------------------------------")
        print("Save Data Dir is %s! " %self.SaveDir)
        self.m_iter_time = m
        self.n_iter_time = n
        print("------------------------------------------------------------")
        print("The Sub-Iterations Times m is %d, n is %d!" %(self.m_iter_time,self.n_iter_time))
        print("------------------------------------------------------------")
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
        self.userPos_ratings_itemPos = saveData["userPos_ratings_itemPos"]
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
        #  FILENAME = "/home/Colin/txtData/forWITFs/Y_n_dic.txt"
        #  self.Y_n_dic = load_from_txt(FILENAME)
        self.Y_n_dic = saveData["Y_n"]



    def main_proceduce(self,iter_num=50,userCount=6682):
        """
            main_proceduce for WITF_Iterations class
            1. add_noises(): add noise for each user
            2. assign_weights() assgin weights by a specific weights configurations
                2.1 How to decide the weights, now set as all 1
            3. Iterations (How to decided whether is convergences??????)
               3.1 Sub-Iterations m (1~5): m=1
               3.2 Sub-Iterations n (1~5): n=1
        """
        # 1. 添加噪声
        self.add_noises()
        print("Finished add_noises()!")
        print("------------------------------------------------------------")
        FBNorm_li = [ ]
        # 计算初始数据的 FBNorm
        ForBe_Norm = self.cal_ObjFunc() 
        #  ForBe_Norm = 0
        FBNorm_li.append(ForBe_Norm)
        print("Start --> Doing Iterations!!!")
        print("------------------------------------------------------------")
        pbar_iter_times = tqdm(range(iter_num))
        for iter_times in pbar_iter_times:
        #  for iter_times in range(iter_num):
            pbar_iter_times.set_description("Each Iteration : ")
            #  print("Start --> Iteration Times : %d" %iter_times)
            # 3. 执行每次迭代，Do each time Iteration
            self.sub_iterations_UVC(userCount)
            # 4. 计算每次的迭代之后的 FBNorm
            ForBe_Norm = self.cal_ObjFunc() 
            FBNorm_li.append(ForBe_Norm)
            #  filename = "/home/Colin/txtData/IterSaves_Pk20_mn1_R15/No" + str(iter_times) + "_iteration.txt"
            #  filename = "/home/Colin/txtData/testTqdm50_newR30/No" + str(iter_times) + "_iteration.txt"
            filename = self.SaveDir + "/No" + str(iter_times) + "_iteration.txt"
            self.save_Data(filename,ForBe_Norm,iter_times)
            #  print("Finished --> Iteration Times : %d" %iter_times)
            #  print(" -----------------------------------------------------------")
            #  print(" -----------------------------------------------------------")
        pbar_iter_times.close()
        #  FBNorm_li_filename = "/home/Colin/txtData/testTqdm50_newR30/FBNorm_li_newDatasets.txt"
        FBNorm_li_filename = self.SaveDir + "/FBNorm_li_newDatasets.txt"
        #  FBNorm_li_filename = "/home/Colin/txtData/IterSaves_Pk20_mn1_R15/FBNorm_li_newDatasets.txt"
        save_to_txt(FBNorm_li,FBNorm_li_filename)
        print("Finished main_proceduce() !")
        print("------------------------------------------------------------")
        return FBNorm_li

    def main_proceduce_old(self,iter_num=5,userCount=6682):
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
        FBNorm_li = [ ]
        ForBe_Norm = self.cal_ObjFunc() 
        FBNorm_li.append(ForBe_Norm)
        for iter_times in range(iter_num):
            self.sub_iterations_UVC(userCount)
            ForBe_Norm = self.cal_ObjFunc() 
            FBNorm_li.append(ForBe_Norm)
        return FBNorm_li

    def assign_weights(self):
        """
            function to assgin weights to irregular tensor
        """
        # now set as 1,1,1,1,1
        pass
        return True

    def sub_iterations_drafts(self,user_Count,m=1,n=1):
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
        pbar_U_cate = tqdm(range(num_K))
        #  for k in range(num_K):
        for k in pbar_U_cate:
            pbar_U_cate.set_description("Update U -- Cate :")
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
            #  print("**Upate U** : In Formula 23 part1 : Done Cate_index:%d !" %k)
        pbar_U_cate.close()
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
        pbar_C_user = tqdm(range(num_N))
        #  for i in range(num_N):
        for i in pbar_C_user:
            pbar_C_user.set_description("Update C --> User :")
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
            #  print("**Upate C** : In Formula 24 part1 : Done UserPos:%d !" %i)
        pbar_C_user.close()
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
                    #  print("get_Y_n: Done userPos:%d,VPos:%d,CPos:%d!" %(u,v,c))
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
        

    def sub_iterations_UVC(self,user_Count=6682,m=3,n=3):
        """
            Functions for do the sub_iterations
        """
        # The number of categories
        m = self.m_iter_time
        n = self.n_iter_time
        num_K = len(self.cate_list)
        num_N = user_Count
        #  num_N = len(self.userPos_li)
        # Pre-Computing Parts
        # sun iteration 1 : update U_i for each user
        pbar_subIter_m = tqdm(range(m))
        #  for m_iter_time in range(m):
        for m_iter_time in pbar_subIter_m:
            pbar_subIter_m.set_description("Sub-Iteration m ( Update U and C ) :")
            C_khaRao_V = self.cal_Khatri_Rao(self.C_Mats,self.V_Mats)
            V_khaRao_U = self.cal_Khatri_Rao(self.V_Mats,self.U_Mats)
            CtCVtV_I = self.cal_AtA_BtB_I(self.C_Mats,self.V_Mats)
            VtVUtU_I = self.cal_AtA_BtB_I(self.V_Mats,self.U_Mats)
            pbar_m_userPos = tqdm(range(num_N))
            #  for user_pos in range(num_N):
            for user_pos in pbar_m_userPos:
                pbar_m_userPos.set_description("SubIter m --> Update U (user):")
                # update row(user_pos) of U by formula 23
                U_i_npa = self.Formula_Ui_23_PreCom(num_K,user_pos,C_khaRao_V,CtCVtV_I)
                self.U_Mats[user_pos,:] = U_i_npa
                #  print("Update U_%d !" %user_pos)
            pbar_m_userPos.close()
            #  print("Finished Update U_i !")
            pbar_m_cate = tqdm(range(len(self.cate_list)))
            #  for cate_index in range(len(self.cate_list)):
            for cate_index in pbar_m_cate:
                pbar_m_cate.set_description("SubIter m --> Update C (cate):")
                # update row(cate_index) of C by formula 24
                num_N = user_Count
                #  num_N = 1000
                C_k_npa = self.Formula_Ck_24_PreCom(num_N,cate_index,V_khaRao_U,VtVUtU_I)
                self.C_Mats[cate_index,:] = C_k_npa
                #  print("Update C_%d !" %cate_index)
            pbar_m_cate.close()
            # Update the whole V by formula 25
            self.Update_V(num_K,num_N)
        # sub iteration 2: Update Pk for each domian k using formula 22
        pbar_subIter_m.close()
        # -------------------------------------------------------------------------------
        pbar_subIter_n = tqdm(range(n))
        #  for n_iter_time in range(n):
        for n_iter_time in pbar_subIter_n:
            pbar_subIter_n.set_description("Sub-Iteration n ( Update Pk ) :")
            # 在此更新Pk,可以使用其他WITF文件里的方法
            self.Update_Pk()
        #  print("Finished Sub-Iteration(New Version) with UserCount:%d !" %user_Count)
        pbar_subIter_n.close()
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

    def analy_trainSets(self):
        """
            Analysis the distributions of the training datasets
        """
        user_ratings_counts = { }
        trainMatrixs = self.training_sparMats_dic["matrix"]
        for cateID in trainMatrixs.keys():
            matrix = trainMatrixs[cateID]
            userIdx = matrix.nonzero()[0]
            itemIdx = matrix.nonzero()[1]
            for pos in range(len(userIdx)):
                #  try:
                    #  userID = userIdx[pos]
                #  except KeyError:
                    #  continue
                itemID = itemIdx[pos]
                userID = userIdx[pos]
                
                if userID in user_ratings_counts:
                    try:
                        user_ratings_counts[userID][cateID] += 1
                    except KeyError:
                        user_ratings_counts[userID][cateID] = 1
                else:
                    user_ratings_counts[userID] = { }
                    user_ratings_counts[userID][cateID] = 1
        for userID in user_ratings_counts.keys():
            user_count = user_ratings_counts[userID]
            Sum = 0
            for cateID in trainMatrixs.keys():
                try:
                    Sum = Sum + user_count[cateID] 
                except KeyError:
                    user_ratings_counts[userID][cateID] = 0
            user_ratings_counts[userID]["total"] = Sum    
        return user_ratings_counts

    def Mats_to_vecA(self,mats):
        """
            Calculate the vec(A) of a matrix A, and return the vec(A)
            vec(A) is a matrix
        """
        col_num = mats.shape[1]
        vecA = mats.getcol(0)
        for col in range(1,col_num):
            col_mat = mats.getcol(col)
            vecA = scipy.sparse.vstack([vecA,col_mat])
        return vecA

    def vecA_to_Mats(self,vecA):
        """
            Calculate the matrix A's vecA
            return the vecA
        """
        row_len = vecA.shape[0]
        row_mats = int(math.sqrt(row_len))
        col_mats = row_mats
        vecA_list = vecA.toarray()
        new_li = [ ]
        new_li.append([ ])
        new_li_inx = 0
        inx_count = 0
        for ele in vecA_list:
            inx_count += 1
            ele = ele[0]
            new_li[new_li_inx].append(ele)
            if inx_count == row_mats:
                inx_count = 0
                new_li_inx += 1
                new_li.append([ ])
            else:
                pass
        col_mat_list = [ ]
        row = [ ]
        col = [ ]
        for i in range(row_mats):
            row.append(i)
            col.append(0)
        #  print(new_li)
        #  print(row)
        #  print(col)
        for idx in range(len(new_li)-1):
            #  row = [0,1,2,3,4]
            #  col = [0,0,0,0,0]
            ele_arr = new_li[idx]
            col_mat = coo_matrix((ele_arr, (row,col)))
            col_mat_list.append(col_mat)
        col_mats_lens = len(col_mat_list)
        Mats = col_mat_list[0]
        for idx in range(1,col_mats_lens):
            Mats = scipy.sparse.hstack([Mats,col_mat_list[idx]])
        return Mats

    def Update_V(self,K_num=5,N_num=6682):
        """
            Update the V as a whole
        """
        C = self.C_Mats
        U = self.U_Mats
        C_rhaRao_U = self.cal_Khatri_Rao(C, U)
        Y2 = self.Y_n_dic["Y_2"]
        Y2_mats = np.asmatrix(Y2)
        Y2_mats = coo_matrix(Y2_mats)
        Y2CU = Y2_mats.dot(C_rhaRao_U)
        vecY2CU = self.Mats_to_vecA(Y2CU)
        # inverse Parts ---------------------------------
        CtC = (C.T).dot(C)
        UtU = (U.T).dot(U)
        CtCUtU = CtC.multiply(UtU)
        size = CtCUtU.shape[0]
        I = SM_identity(size)
        CtCUtU_I = CtCUtU + I
        CtCUtU_kron_I = SM_kron(CtCUtU_I,I)
        #  print("Finished CtCUtU_kron_I !")
       	# print(CtCUtU_kron_I)
        inv_part = CtCUtU_kron_I + self.Update_V_subpart1(K_num,N_num)
        #  print("Finished SubParts !")
        inv_value = SSL_inv(inv_part.tocsc()) 		
        vecV = inv_value.dot(vecY2CU)
        #  print("Finished Inverse Operations")
        V = self.vecA_to_Mats(vecV)
        #print(V)
        self.V_Mats = V
        return True

    def Update_V_subpart1(self,K=5,N=6682):
        """
            Calate the SUMMARY K,N Parts
        """
        U = self.U_Mats
        pbar_V_cate = tqdm(range(K))
        #  for k in range(K):
        for k in pbar_V_cate:
            pbar_V_cate.set_description("Update V -- Each Cate:")
            cateID = self.cate_list[k]
            P_k = self.P_k_dic[cateID]
            P_k_t = P_k.T
            C_k = self.C_Mats.getrow(k).toarray()[0]
            sigma_k = SM_diags(C_k)
            W_kij = self.ratings_weights_matrixs_dic[cateID]
            res = 0
            pbar_V_user = tqdm(range(N))
            #  for i in range(N):
            for i in pbar_V_user:
                pbar_V_user.set_description("Update V -- Each User:")
                U_i = U[i]
                U_i_t = U_i.T
               	omiga_ki = self.Wkij_dic[cateID][i]
                Sk_Uit = sigma_k.dot(U_i_t)
                Sk_Uit_Ui = Sk_Uit.dot(U_i)
                Sk_Uit_Ui_Sk = Sk_Uit_Ui.dot(sigma_k)
                # Pkt_Oki_Pk parts
                Pkt_Oki = P_k_t.dot(omiga_ki)
                Pkt_Oki_Pk = Pkt_Oki.dot(P_k)
                res = res + SM_kron(Sk_Uit_Ui_Sk,Pkt_Oki_Pk)
                #  print("**Upadte_V** : Finished CateID: %d, UserID: %d !" %(cateID,i))
            pbar_V_user.close()
        pbar_V_cate.close()
        return res

    def Update_Pk(self):
        """
            Calculate the constrant Pk for each category by SVD
        """
        cate_list = self.cate_list
        pbar_Pk_cate = tqdm(range(len(cate_list)))
        #  for cate_index in range(len(cate_list)):
        for cate_index in pbar_Pk_cate:
            pbar_Pk_cate.set_description("Update Pk --> cate:")
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
            #  print("**Update Pk** : Finished Update_Pk in cateID:%d !" %cateID)
        #  print("Finished Update_Pk!")
        pbar_Pk_cate.close()
        return True

    def cal_ObjFunc(self):
        """
            calculate the Forbenius Norm values for the objective functions
            return the Forbenius Norm value
        """
        U = self.U_Mats
        V = self.V_Mats
        fro_norm_value = 0
        for k in range(len(self.cate_list)):
            # calculate in each category
            cateID = self.cate_list[k]
            X_k = self.training_sparMats_dic["matrix"][cateID]
            C_k_row = self.C_Mats.getrow(k).toarray()[0]
            Sigma_k = SM_diags(C_k_row)
            P_k = self.P_k_dic[cateID]
            USk = U.dot(Sigma_k)
            PkVt = (P_k.dot(V)).T
            MAT = X_k - USk.dot(PkVt)
            fro_norm_value += 0.5 * (SSL_ForNorm(MAT))**2
        #  print("Finished calculate the FroBenius Norm for the Obejective Functions!")
        return fro_norm_value

    def save_Data(self,filename,ObjVal,iter_times):
        """
            save files for each time iterations
        """
        saved_data = { }
        saved_data["objValue"] = ObjVal
        saved_data["IterTimes"] = iter_times
        saved_data["testSets"] = self.test_data_dic
        saved_data["trainSets"] = self.training_sparMats_dic
        saved_data["userPos"] = self.userPos_li
        saved_data["itemPos"] = self.itemPos_dic
        saved_data["U"] = self.U_Mats
        saved_data["V"] = self.V_Mats
        saved_data["C"] = self.C_Mats
        saved_data["Pk"] = self.P_k_dic
        saved_data["target_cateID"] = self.target_cateID
        save_to_txt(saved_data,filename)
        return True

    def cal_users_noisePos(self):
        """
            The function to calculate the noises postions for each user in each category
            Just for one-time-iteration!!!
        """
        pass
        return True


# ------------------------------------------------------------------------------------------------------
# main functions
# ------------------------------------------------------------------------------------------------------
U = 10
I = 10 
init_left = 20
init_right = 30
TC = 40
R = 5
UserNumbers = 2403
IterTimes = 20
mn = 3
txtfile = "/home/Colin/GitHubFiles/U" + str(U) + "I" + str(I) + "_PreCom_Data/R" + str(R) + "_init" + str(init_left) + "to" + str(init_right) + "_U" + str(U) + "I" + str(I) + "_TC" + str(TC) + "_preCom_Data/new_WITF_precomputed_Data.txt"
#  txtfile = "/home/Colin/GitHubFiles/U10I10_PreCom_Data/R5_init1to5_U10I10_TC17_preCom_Data/new_WITF_precomputed_Data.txt"
savedir = "/home/Colin/txtData/U" + str(U) + "I" + str(I) + "_Iterated_Data/R" + str(R) + "_init" + str(init_left) + "to" + str(init_right) + "_U" + str(U) + "I" + str(I) + "_TC" + str(TC) + "_mn" + str(mn) + "_Iter" + str(IterTimes) 
# print(txtfile)
# print(savedir)
#txtfile = "/home/Colin/txtData/forWITFs/WITF_Pre_Computed_Data.txt"
IWITF = WITF_Iterations(txtfile,savedir,mn,mn)
print("Created the instant of WITF_Iterations class which named IWITF!")
starttime = datetime.datetime.now()
#  IWITF.main_proceduce(20,50)
#  IWITF.main_proceduce(2,100)
IWITF.main_proceduce(IterTimes,UserNumbers)
endtime = datetime.datetime.now()
executetime = (endtime - starttime).seconds
print("Finished All !!!!, and the Execute Time is %d" %executetime)


# --------- end lines -------------------
#  IWITF.sub_iterations(100)
#  IWITF.sub_iterations_UVC(1000)
# IWITF.sub_iterations_UVC(100)


