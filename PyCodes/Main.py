# Python3
# Filename: Main.py
# Usages : WITF Main Functions

from tqdm import tqdm
import datetime
import random
import math
import copy
import numpy as np
import scipy
import os

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
from WITF import *
from WITF_Iterations import *


def OS_mkdir(path):
    # 引入模块

    # 去除首位空格
    # path=path.strip()
    # 去除尾部 \ 符号
    # path=path.rstrip("\\")

    # 存在     True
    # 不存在   False
    print("----------------Target Dir-------------------------")
    #  print(path)
    isExists=os.path.exists(path)

    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print("***********************************************************")
        print("***********************************************************")
        print(path)
        print("Dir creating Successfully!")
        print("***********************************************************")
        print("***********************************************************")
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print("***********************************************************")
        print("***********************************************************")
        print(path)
        print("Dir Already Existed!")
        print("***********************************************************")
        print("***********************************************************")
        return True
    return False

# -------------------------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Constants Definitions
# -------------------------------------------------------------------------------------------------
U = 10
I = 10 
UserNumbers = 2403
# Init Values Range For Matrix U,V,C
init_left = 20
init_right = 30
# Sub-iteration Times in each Iteration
mn = 3
# Target Category ID
TC = 4
# R numbers : but now there is no usages, just for create files and dir names
R = 5
# Iteration Times 
Iter_Times = 20
IterTimes = Iter_Times
# Add noises count for each crawler user
Noise_Count_PerIter = 2 

# -------------------------------------------------------------------------------------------------
# File Name Parts
# -------------------------------------------------------------------------------------------------

# raw_data_file : Raw data calculated by TensorIrr.py, It needs to be calculated by WITF.py
raw_data_file = "/home/Colin/GitHubFiles/new_WITF_data/Raw_Datasets/User" + str(U) + "_Item" + str(I) + "/new_raw_data_for_WITF_py.txt"

# SaveDir for PreComputed Data
fileDir = "/home/Colin/GitHubFiles/U" + str(U) + "I" + str(I) + "_PreCom_Data/Revised_WITF/R" + str(R) + "_init" + str(init_left) + "to" + str(init_right) + "_U" + str(U) + "I" + str(I) + "_TC" + str(TC) + "_preCom_Data"
# Check this Dir
OS_mkdir(fileDir)

# SaveFile for PreComputed Data
filename = fileDir + "/new_WITF_precomputed_Data.txt"

# PreComputed Data Files 
txtfile = filename

# Results Data SaveDir
savedir = "/home/Colin/Revised_WITF_Results/U" + str(U) + "I" + str(I) + "_Iterated_Data/R" + str(R) + "_init" + str(init_left) + "to" + str(init_right) + "_U" + str(U) + "I" + str(I) + "_TC" + str(TC) + "_mn" + str(mn) + "_Iter" + str(IterTimes) 
# Check this Dir
OS_mkdir(savedir)

# -------------------------------------------------------------------------------------------------
# WITF PreComputed Parts
# -------------------------------------------------------------------------------------------------

# WITF ProComputed Data : __init__
witf = WITF(raw_data=raw_data_file,SaveFile=filename,R_latent_feature_Num=R,target_cateID=TC,init_range=(init_left,init_right),noiseCount=Noise_Count_PerIter,add_noise_times=Iter_Times)
# WITF ProComputed Data : Mian Procedures
witf.main_proceduce()

# -------------------------------------------------------------------------------------------------
# WITF Iterations Parts
# -------------------------------------------------------------------------------------------------

# WITF Iterations : __init__
IWITF = WITF_Iterations(txtfile,savedir,mn,mn)

starttime = datetime.datetime.now()

# WITF main_proceduce 函数，程序的主流程，即算法的 Iteration 部分
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
IWITF.new_main_proceduce(IterTimes,UserNumbers)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
endtime = datetime.datetime.now()
executetime = (endtime - starttime).seconds
print("Finished All !!!!, and the WITF Iterations Execute Time is %d" %executetime)

# -------------------------------------------------------------------------------------------------
# END 
# -------------------------------------------------------------------------------------------------
