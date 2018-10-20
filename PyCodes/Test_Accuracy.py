# Python3
# Filename: WITF.py
# Usages : WITF model codes

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

class Test_Accuracy:
    ''' Class for test the accuracy for WITF model '''
    def __init__(self,filename,it_No):
        self.it_no = it_No
        self.data = load_from_txt(filename)
        # Keys: objValue, IterTimes,testSets,trainSets, U,V,C
        self.testDataDic = self.data["testSets"]["dataDic"]
        self.U = self.data["U"]
        self.C = self.data["C"]
        self.V = self.data["V"]
        self.Pk = self.data["Pk"]
        cate_index = 4
        C_k_row = self.C.getrow(cate_index).toarray()[0]
        Sigma_k = SM_diags(C_k_row)
        self.Uk = Sigma_k.dot((self.U).T)
        Pk = self.Pk[4]
        self.Vk = (Pk.dot(self.V)).T
        self.realPre_li = [ ]


    def test_for_all_users_MAE(self):
        """
            Test the accuracy by using the RMSE matircs
        """
        realPre_li = [ ]
        for key in self.testDataDic.keys():
            real_rating = self.testDataDic[key]
            userID = key[0]
            itemPos = key[1]
            pre_rating = self.cal_preRating(userID,itemPos)
            realPre_li.append((real_rating,pre_rating))
            self.realPre_li.append((real_rating,pre_rating))
        print("In Iteration %d --> Calculate Prediction Ratings Done!" %self.it_no)
        print("MAE Test : The real&Prediction ratings are as below: ")
        print(self.realPre_li)
        SUM = 0
        for ele in realPre_li:
           ABS = abs(ele[0] - ele[1])
           SUM = SUM + ABS
           #  sqrt_eleAbs = (ele[0]-ele[1])**2
           #  SUM = SUM + sqrt_eleAbs
        MAE = SUM/(len(self.testDataDic))
        print("In Iteration %d --> Calculate MAE Done!" %self.it_no)
        return MAE 

    def test_for_all_users_RMSE(self):
        """
            Test the accuracy by using the RMSE matircs
        """
        realPre_li = [ ]
        for key in self.testDataDic.keys():
            real_rating = self.testDataDic[key]
            userID = key[0]
            itemPos = key[1]
            pre_rating = self.cal_preRating(userID,itemPos)
            realPre_li.append((real_rating,pre_rating))
            self.realPre_li.append((real_rating,pre_rating))
        print("In Iteration %d --> Calculate Prediction Ratings Done!" %self.it_no)
        print("RMSE Test : The real&Prediction ratings are as below: ")
        print(self.realPre_li)
        SUM = 0
        for ele in realPre_li:
           sqrt_eleAbs = (ele[0]-ele[1])**2
           SUM = SUM + sqrt_eleAbs
        RMSE = math.sqrt(SUM/(len(self.testDataDic)))
        print("In Iteration %d --> Calculate RMSE Done!" %self.it_no)
        return RMSE 

    def cal_preRating(self,userID,itemPos):
        """
            calculate the  predict ratings
        """
        #  cate_index = 4
        #  C_k_row = self.C.getrow(cate_index).toarray()[0]
        #  Sigma_k = SM_diags(C_k_row)
        #  Uk = Sigma_k.dot((self.U).T)
        #  Pk = self.Pk[4]
        #  Vk = (Pk.dot(self.V)).T
        Uk_i = self.Uk.getcol(userID)
        Vk_j = self.Vk.getcol(itemPos)
        rating_mat = (Uk_i.T).dot(Vk_j)
        preRating = rating_mat.toarray()[0][0]
        return preRating

# ---------------------------------------------------------------------
#    Main Functions Parts
# ---------------------------------------------------------------------

#  filename = "/home/Colin/txtData/IterSaves_Pk20/No4_iteration.txt"
#  Data = Test_Accuracy(filename)
#  RMSE = Data.test_for_all_users()
RMSE_list = [ ]
MAE_list = [ ]
DATA = [ ]
for i in range(5):
    filename = "/home/Colin/txtData/IterSaves_Pk20_mn1_R15/No" + str(i) + "_iteration.txt"
    #  filename = "/home/Colin/txtData/IterSaves_Pk50_mn3/No" + str(i) + "_iteration.txt"
    #  filename = "/home/Colin/txtData/IterSaves_Pk50/No" + str(i) + "_iteration.txt"
    Data = Test_Accuracy(filename,i)
    RMSE = Data.test_for_all_users_RMSE()
    RMSE = Data.test_for_all_users_MAE()
    RMSE_list.append(RMSE)
    MAE_list.append(RMSE)
    DATA.append(Data)

print("All RMSE are as below: ")
print(RMSE_list)
print("All MAE are as below: ")
print(MAE_list)
print("Finished All!!!")
