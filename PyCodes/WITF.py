# Python3
# Filename: WITF.py
# Usages : WITF model codes

from tqdm import tqdm
import random
import numpy as np
import scipy
import copy

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
    
    def __init__(self,raw_data,SaveFile,R_latent_feature_Num=5,init_range=(1,6),target_cateID=4,ratios=0.8,noiseCount=2,add_noise_times=5):
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
        #  self.R_latent_feature_Num = R_latent_feature_Num
        #  print("R is %d!" %self.R_latent_feature_Num)
        self.raw_data_file = raw_data
        print("RawDataFile is %s!" %self.raw_data_file)
        self.SaveFile = SaveFile
        print("SaveFile is %s!" %self.SaveFile)
        self.init_range = init_range
        print("Randomly Init Range is %d ~ %d!" %(self.init_range[0],self.init_range[1]))
        self.Wkij_dic = {}
        # ****************************************************************************
        # ****************************************************************************
        # 读取由 TensorIrr产生的数据(txt files)
        # self.raw_data.keys() : userPos, itemPos, matrix, ratingsPos
        self.raw_data = load_from_txt(self.raw_data_file)
        #  load_from_txt("/home/Colin/GitHubFiles/new_WITF_data/Raw_Datasets/User10_Item10/new_raw_data_for_WITF_py.txt")
        # load_from_txt("/home/Colin/txtData/forWITFs/WITF_raw_data_5_domains.txt")
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
        # 计算 R 的值，(Latent Feature Numbers)
        users_count = 0 # 用户总数
        ratings_count = 0 # 所有的评分
        for cateID in self.raw_sparMats_dic.keys():
            cate_raw_mats = self.raw_sparMats_dic[cateID]
            users_count = (cate_raw_mats.shape)[0]
            ratings_count += len(cate_raw_mats.nonzero()[0])
        R = ratings_count//users_count + 1 
        self.R_latent_feature_Num = R
        print("R is %d!" %self.R_latent_feature_Num)
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
        # for save Y_n
        self.Y_n_dic = { }

    def main_proceduce(self):
        """
           The main_proceduce for the WITF model
           Parameter : target_cateID --> choose which category as the target category 
                       train_data_ratio --> set the ratio(%) for the training data and test data
        """
        # 1. 按照指定的比例ratios把所有数据分割成trainingSet和testSet
        self.split_data_byRatios()
        print("Finished SPLIT training and test dataset with target_cateID: %d and Ratios: %d Precentages!" \
                 %(self.target_cateID, 100*self.ratios))
        # 2. 计算训练集的stats, 在迭代更新时直接使用，减少迭代更新过程的计算时间
        self.cal_stats_for_trainSets()
        print("Finished calculate the stats for trianing datasets!")
        # 3. 为原始数据（训练集）计算添加noises(Virtual Data)的位置，为了减少迭代的计算时间
        # ----> STEP:4 --- Prepare for STEP: 5 (In WITF_Iterations Class)
        self.cal_users_noisePos()
        # 4. 初始化，U，V，C 矩阵，所有值都在（0，1）之间
        # ----> STEP 1(V <- I) & 2
        self.randomly_init_U_C_V()
        print("Finshed randomly_init_U_C_V() Functions")
        # 5. 计算 {Pk}
        # ----> STEP 3
        self.find_Pk()
        print("Finshed find_Pk() Functions")
        # 6. 初始化 weights over ratings （w_kij)
        self.init_ratings_weights_matrix()
        print("Finished init_ratings_weights_matrix() Functions")
        # 7. 设置 weights over ratings （w_kij), 有rating为1，否则为0
        self.set_observation_weights()
        print("Finished set_observation_weights() Functions")
        # 8. 事先计算Wki,减少迭代的计算时间
        self.cal_Wki()
        print("Finished cal_Wki() Functions")
        # 9, 存储Precomputed的所有数据
        # ---> 包含计算 Y_n_dic，事先计算的数据，减少迭代的计算时间
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
        saveData["omiga_ki"] = self.Wkij_dic
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
        self.get_Y_n()
        saveData["Y_n"] = self.Y_n_dic
        #  filename = "/home/Colin/GitHubFiles/new_WITF_data/R5_init1to100_preCom_Data/new_WITF_precomputed_Data.txt"
        #  filename = "/home/Colin/GitHubFiles/new_WITF_data/new_WITF_precomputed_Data.txt"
        #filename = "/home/Colin/txtData/forWITFs/WITF_Pre_Computed_Data.txt"
        save_to_txt(saveData,self.SaveFile)
        return True

    def find_Pk(self):
        """
            Calculate the constrant Pk for each category by SVD
        """
        catelist = self.training_sparMats_dic["matrix"].keys()
        cate_list = [ ]   # cate_list = [4,17,24,29,40]
        for key in catelist:
            cate_list.append(key)
        for cate_index in range(len(catelist)):
        #  for cateID in self.training_sparMats_dic["matrix"]:
            cateID = cate_list[cate_index] # cateID = 4,17,24,29,40
            # Raw data matrix (a cate) X_k
            X_k = self.training_sparMats_dic["matrix"][cateID]
            C_k_row = self.C_Mats.getrow(cate_index).toarray()[0]
            Sigma_k = SM_diags(C_k_row)
            # SVD Operations
            SVD_mats = X_k.T
            SVD_mats = SVD_mats.dot(self.U_Mats)
            SVD_mats = SVD_mats.dot(Sigma_k)
            SVD_mats = SVD_mats.dot(self.V_Mats.T)
            # SVD R select questions??
            #  A_r, sigma_r, B_r_t = SSL_svds(SVD_mats)
            A_r, sigma_r, B_r_t = SSL_svds(SVD_mats,k=self.R_latent_feature_Num-1)
            P_k = coo_matrix(A_r.dot(B_r_t))
            self.P_k_dic[cateID] = P_k
            print("Found P_k for CateID: %d!" %cateID)
        print("Finished find all {P_k}!")
        return True


    def randomly_init_U_C_V(self):
        """
            randomly init the latent feature matrices: U, C, V
            所有的随机初始值都在（0，1）之间，存在疑问（如何更好的设置随机初始值）？？
        """
        randint_left = self.init_range[0]
        randint_right = self.init_range[1]
        U_N_userNum = len(self.userPos_li)
        R = self.R_latent_feature_Num
        C_K_cateNum = 5
        # init V as scipy sparse identity matrix
        self.V_Mats = SM_identity(R)
        print("Finished init V_mats! (Identity Matrix)")                
        # Randomly init U, C with density = 1?
        self.U_Mats = SM_random(U_N_userNum,R,density=1,format='dok')
        for i in range(U_N_userNum):
            for j in range(R):
                self.U_Mats[i,j] = np.random.randint(randint_left,randint_right)
        print("Finished init U_mats!")                
        self.C_Mats = SM_random(C_K_cateNum,R,density=1,format='dok')
        for i in range(C_K_cateNum):
            for j in range(R):
                self.C_Mats[i,j] = np.random.randint(randint_left,randint_right)
        print("Finished init C_mats!")                
        return True

    def cal_users_noisePos(self):
        """
            The function to calculate the noises postions for each user in each category
        """
        print("Start --> cal_users_noisePos")
        pbar_lv1 = tqdm(self.userPos_ratings_itemPos.keys())
        #  for user_pos in self.userPos_ratings_itemPos:
        #  user_pos 为按 userID 的排列顺序，user_pos = [0,1,2,......]
        for user_pos in pbar_lv1:
            # 遍历每一个用户
            pbar_lv1.set_description("Each User:")
            self.user_noisePos[user_pos] = { }
            ratingPos_dic = self.userPos_ratings_itemPos[user_pos]
            pbar_lv2 = tqdm(ratingPos_dic.keys())
            #  for cateID in ratingPos_dic:
            for cateID in pbar_lv2:
                # 遍历每个用户的每一个分类，查看都是那些item被评价了
                pbar_lv2.set_description("Each Cate:")
                self.user_noisePos[user_pos][cateID] = [ ]
                # 提取该用户在该分类的所有评价的item (itemPos) , 注意非ItemID
                rating_cateID = ratingPos_dic[cateID]
                if rating_cateID[0] is False:
                    # 如果该用户在该分类没有评分
                    for time in range(self.add_noise_times):
                        allItemPos_dic = self.trainSets_stats_dic[cateID]["allItemPos"]
                        try:
                            selectedPos = random.sample(allItemPos_dic.keys(),self.noiseCount)
                            #  selectedPos = random.sample(self.trainSets_stats_dic[cateID]["allItemPos"].keys(),self.noiseCount)
                        except ValueError:
                            selectedPos = self.trainSets_stats_dic[cateID]["allItemPos"].keys()
                            # return False
                            # selectedPos = []
                        self.user_noisePos[user_pos][cateID].append(selectedPos)
                else:
                    userPos_rating_cateID = self.userPos_ratings_itemPos[user_pos][cateID][1]
                    all_itemPos_cateID_dic = self.trainSets_stats_dic[cateID]["allItemPos"]
                    #  for itemPos in userPos_rating_cateID:
                        #  try:
                            #  del all_itemPos_cateID_dic[itemPos]
                        #  except KeyError:
                            #  pass
                    all_itemPos_cateID_dic = self.Drafts_del_dict(userPos_rating_cateID,all_itemPos_cateID_dic)    
                    blankPos = sorted(all_itemPos_cateID_dic.keys())
                    for time in range(self.add_noise_times):
                        if len(blankPos) > self.noiseCount:
                            selectedPos = random.sample(blankPos,self.noiseCount)
                        else:
                            selectedPos = blankPos
                        self.user_noisePos[user_pos][cateID].append(selectedPos)
                #  print("Finished calculate %d group users noises postions for userPos:%d in cateID:%d!" %(self.add_noise_times,user_pos,cateID)) 
            #  print("Finished userPos:%d <--------------------> " %user_pos)
            pbar_lv2.close()
        pbar_lv1.close()
        print("Finsihed --> cal_users_noisePos")
        return True

    def Drafts_del_dict(self,Arr,B_dic):
        deep_B = copy.deepcopy(B_dic)
        for ele in Arr:
            try:
                del deep_B[ele]
            except KeyError:
                pass
        return deep_B

    def add_noises(self):
        """
            add noises(virtual data) for each user in each category
            主要在迭代类中使用，这里只是草稿，备用
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
            主要在迭代类中使用，这里只是草稿，备用
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
        """
            计算训练集的stats，包括：mean,mu_k,sigma等重要数据，以便在进行迭代更新
            的时候进行直接使用，减少迭代的计算时间
            主要是为了添加噪声的时候方便使用
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        """
        print("Start --> cal_stats_for_trainSets")
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
            # ???????????????????? 作用未知，忘记了！！！
            itemPos_dic = { }
            for itemPos in range(itemCount):
                #  itemPos_li.append(itemPos)
                itemPos_dic[itemPos] = True
            #  self.cateID_itemPos_dic[cateID] = itemPos_dic
            self.trainSets_stats_dic[cateID]["allItemPos"] = itemPos_dic
            #  self.all_blank_pos_dic[cateID] = itemPos_li
        print("Finished --> cal_stats_for_trainSets")
        return True

    def split_data_byRatios(self):
        """
            split the training data and test data by ratio and target_cateID
        """
        print("Start --> split_data_byRatios !")
        target_cateID = self.target_cateID
        ratios = self.ratios
        self.test_data_dic["dataDic"] = { }
        self.training_sparMats_dic["target_cateID"] = target_cateID
        self.training_sparMats_dic["ratio"] = ratios
        self.training_sparMats_dic["matrix"] = self.raw_sparMats_dic.copy()
        self.test_data_dic["target_cateID"] = target_cateID
        self.test_data_dic["ratio"] = ratios
        print("Target_Cate is %d!" %target_cateID)
        target_sparMat = self.raw_sparMats_dic[target_cateID]
        # 找到所有的非零的 row_index and col_index
        row_NonZero = target_sparMat.nonzero()[0]
        col_NonZero = target_sparMat.nonzero()[1]
        # 进行分割抽取，抽取位置的选择由 “Drafts_samples"函数决定
        sampled_index = Drafts_samples(len(row_NonZero),1-ratios)
        pbar = tqdm(sampled_index)
        for index in pbar:
        #  for index in sampled_index:
            pbar.set_description("selecting test data(ratings) :")
            # get each randomly select test data's row and col
            row = row_NonZero[index]
            col = col_NonZero[index]
            rating = target_sparMat[row,col]
            # the randomly select data is deleted(changed to 0 ??)
            # 被选取的数据是不是要设置为0，这点需要考虑
            target_sparMat[row,col] = 0
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            self.test_data_dic["dataDic"][(row,col)] = rating
            # update a element into test data
            # self.test_data_dic["datalist"].append((row,col,rating))
        pbar.close()
        self.training_sparMats_dic["matrix"][target_cateID] = target_sparMat
        print("Finished --> split_data_byRatios !")
        return True
    
    def bak_split_data_byRatios(self):
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
        # 进行分割抽取，抽取位置的选择由 “Drafts_samples"函数决定
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
        根据每个Category 的用户数和item数,初始化每个category的weight matirx (w_kij) 
        此weights matrix 主要是用于表示每个rating的权重，属于 weights over ratings not for domains 
        '''
        # users_count : 用户总数
        users_conuts = len(self.userPos_li)  # 以前的用户选择标准
        for cate in self.itemPos_dic:
            # items_counts : 每一个cate的 item 数目
            items_counts = len(self.itemPos_dic[cate][1])
            sm = dok_matrix((users_conuts,items_counts),
                    dtype=np.float32)
            self.ratings_weights_matrixs_dic[cate] = sm
            # 初始化 W_kij 完毕, 所有内部元素（即每一个评分的权重）全部为0
            print("ratings_weights_matrix for category: %d, size is%d and %d!"
                    %(cate,users_conuts,items_counts))
        return True

    def set_observation_weights(self):
        """
            为每个weight matrix(w_kij) 的每个元素（weights for item) 设置值
            观察到的rating 的 weight 设置为 1
            没有观察到的，即缺失的值，设置为 0
        """
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

    def cal_Wki(self):
        """
            事先计算Wki,减少迭代的计算时间
        """
        print("Start --> cal_Wki")
        pbar_lv1 = tqdm(self.ratings_weights_matrixs_dic.keys())
        for cateID in pbar_lv1:
        #  for cateID in self.ratings_weights_matrixs_dic:
            pbar_lv1.set_description("Each Cate:")
            W_kij = self.ratings_weights_matrixs_dic[cateID]
            self.Wkij_dic[cateID] = {}
            pbar_lv2 = tqdm(range(len(self.userPos_li)))
            #  for user_pos in range(len(self.userPos_li)):
            for user_pos in pbar_lv2:
                pbar_lv2.set_description("Each User:")
                W_ki = W_kij.getrow(user_pos).toarray()[0]
                omiga_ki = SM_diags(W_ki)
                size = omiga_ki.shape[0]
                I_omiga_ki = SM_identity(size)
                omiga_ki = omiga_ki - I_omiga_ki
                self.Wkij_dic[cateID][user_pos] = omiga_ki
                #  print("Finished omiga_ki with cateID:%d,userPos:%d !" %(cateID,user_pos))
            pbar_lv2.close()
        pbar_lv1.close()
        print("Finished --> cal_Wki")
        return True        

    def get_Y_n(self):
        """
            function to get tensor Y mode-n unfolding
            事先计算，减少迭代的计算时间
        """
        print("Start --> get_Y_n")
        User_num = len(self.userPos_li)
        Cate_num = 5
        #  Cate_num = len(self.cate_list)
        V_num = self.R_latent_feature_Num
        R_num = self.R_latent_feature_Num
        Y = np.random.rand(User_num,V_num,Cate_num)
        pbar = tqdm(range(User_num))
        for u in pbar:
        #  for u in range(User_num):
            pbar.set_description("Each User:")
            U_u = self.U_Mats.getrow(u)#.toarray[0]
            for v in range(V_num): 
                V_v = self.V_Mats.getrow(v)#.toarray[0]
                for c in range(Cate_num):
                #  for c in range(Cate_num):
                    C_c = self.C_Mats.getrow(c)#.toarray[0]
                    entry = U_u.multiply(V_v)
                    entry = entry.multiply(C_c).sum()
                    Y[u][v][c] = entry
                    #  print("get_Y_n: Done userPos:%d,VPos:%d,CPos:%d!" %(u,v,c))
        self.Y_n_dic["Y_1"] = np.reshape(np.moveaxis(Y,0,0),(Y.shape[0], -1),order='F')
        self.Y_n_dic["Y_2"] = np.reshape(np.moveaxis(Y,1,0),(Y.shape[1], -1),order='F')
        self.Y_n_dic["Y_3"] = np.reshape(np.moveaxis(Y,2,0),(Y.shape[2], -1),order='F')
        pbar.close()
        print("Finished --> get_Y_n")
        return True

# ================================================================================================
#   Global Fucntions
# ================================================================================================

# ================================================================================================
#   Main Fucntions
# ================================================================================================
U = 10
I = 10 
init_left = 1
init_right = 6
TC = 4
R = 5
Iter_Times = 20
Noise_Count_PerIter = 5 

raw_data_file = "/home/Colin/GitHubFiles/new_WITF_data/Raw_Datasets/User" + str(U) + "_Item" + str(I) + "/new_raw_data_for_WITF_py.txt"
#  raw_data_file = "/home/Colin/GitHubFiles/new_WITF_data/Raw_Datasets/User10_Item10/new_raw_data_for_WITF_py.txt"

filename = "/home/Colin/GitHubFiles/U" + str(U) + "I" + str(I) + "_PreCom_Data/Revised_WITF/R" + str(R) + "_init" + str(init_left) + "to" + str(init_right) + "_U" + str(U) + "I" + str(I) + "_TC" + str(TC) + "_preCom_Data/new_WITF_precomputed_Data.txt"
#  filename = "/home/Colin/GitHubFiles/U10I10_PreCom_Data/R5_init50to60_U10I10_TC17_preCom_Data/new_WITF_precomputed_Data.txt"
witf = WITF(raw_data=raw_data_file,SaveFile=filename,R_latent_feature_Num=R,target_cateID=TC,init_range=(init_left,init_right),noiseCount=Noise_Count_PerIter,add_noise_times=Iter_Times)
# ---------------------------------------------------------------------------------
# WITF Class Main Procedures
# ---------------------------------------------------------------------------------
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# witf.main_proceduce()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

witf.split_data_byRatios()
witf.cal_stats_for_trainSets()
witf.cal_users_noisePos()
# 需要认真考虑这个初始化函数 : randomly_init_U_C_V()
witf.randomly_init_U_C_V()
print("Finshed randomly_init_U_C_V() Functions")
witf.find_Pk()
witf.init_ratings_weights_matrix()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

