#! python3
# Filename: irregularTensor.py 
# Usages: defined irregular tensor class


import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook
import scipy.sparse as sparse

#import MySQL_c
from MySQL_c import MySQL


class IrregularTensor:
    '''Class for Irregular Tensor'''
    def __init__(self):
        # initial simple class for the irregular tensor
        self.tensor = {}
        self.mysql_c = MySQL()
        # Connect to Epinions
        self.mysql_c.connect2MySql()
        #user_sum = self.mysql_c.executeSQL("select count(*) from User;")
        self.user_sum = 306368
        product_sum = self.mysql_c.executeSQL("select count(*) from Product;")
        self.product_sum = product_sum[0][0]
        self.products_dic = {}
        self.matrixs_dic = {}

    def UpdateAllTensor(self):
        # Update all irregular tensor
        cmd_s = "SELECT * FROM Category where parent is NULL;"
        # Get all parent categories
        parent_category = self.mysql_c.executeSQL(cmd_s)


    def GetProductListInEachSubCategory(self, idcategory):
        # Return all product ids of a sub parent category(ID)
        cmd_s = "select idproduct from product where idcategory = " + str(idcategory) + ";"
        products = self.mysql_c.executeSQL(cmd_s)
        product_list = []
        for ele in products:
            product_list.append(ele[0])
        return product_list

    def GetProductListInParentCategory(self, parent_category):
        # Return all product ids of a parent category(ID)
        cmd_s = "select idcategory,name from Category where parent = " + str(parent_category) + ";"
        sub_category = self.mysql_c.executeSQL(cmd_s)
        #print(sub_category)
        sub_list = []
        for sub in sub_category:
            sub_list.append(sub[0])
        #print(sub_list)
        all_product_list = []
        for idcategory in sub_list:
            #print(idcategory)
            lis = self.GetProductListInEachSubCategory(idcategory)
            all_product_list.extend(lis)
        np_all_product = np.array(all_product_list)
        np_all_product.sort()
        return np_all_product

    def GetEachParentCategoryProducts(self):
        # Get the products for each parent category (In dictionary)
        parents_products_dic = {}
        cmd_s = "select idcategory,name from category where parent is null;"
        parents_tuple = self.mysql_c.executeSQL(cmd_s)
        for ele_tuple in parents_tuple:
            parent_id = ele_tuple[0]
            #print(parent_id)
            products_in_parent = self.GetProductListInParentCategory(parent_id)
            if len(products_in_parent) is not 0:
                parents_products_dic.update({parent_id : (ele_tuple[1], products_in_parent)})

        return parents_products_dic

    def UpdateProductsInAllDomains(self):
        self.products_dic = self.GetEachParentCategoryProducts()

    def FindParentForProduct_bak(self, product_id):
        for (IDs,category) in self.products_dic.items():
            #print(IDs)
            #print(category)
            result = np.where((category[1] == product_id).any())
            #print(result)
            if len(result[0]) is 1 :
                return IDs
        return False


    def FindParentForProduct(self, product_id):
        for (IDs,category) in self.products_dic.items():
            #print(IDs)
            #print(category)
            result = np.argwhere(category[1] == product_id)

            #print(result)
            if len(result) is 1 :
                return IDs,result[0][0]
        return 0,0

    def UpdateMatrixs(self):
        self.CreateInitMatrix()
        no_parent_counts = 0
        user_out_range = 0
        cmd_s = "Select * from Review;"
        reviews = self.mysql_c.executeSQL(cmd_s)
        for rating in tqdm_notebook(reviews, desc = "Reviews Finished Counts"):
            UserID = rating[1]
            ItemID = rating[4]
            RatingValue = rating[2]*0.1 + ItemID
            #print(ItemID)
            ParentID, Position = self.FindParentForProduct(ItemID)
            #print(ParentID)
            if (ParentID == 0) and (Position == 0):
                no_parent_counts += 1
            elif (UserID<self.user_sum) :
                self.matrixs_dic[ParentID][UserID,Position] = RatingValue
            else:
                user_out_range += 1

        return no_parent_counts, user_out_range
            # self.matrixs_dic[]

    def CreateInitMatrix(self):
        for key in self.products_dic.keys():
            item_counts = len(self.products_dic[key][1])
            #item_IDs = self.products_dic[key][1]
            mtx = sparse.dok_matrix((self.user_sum,item_counts), dtype=np.float16)
            #mtx = sparse.dok_matrix((self.user_sum, item_IDs), dtype=np.float16)
            self.matrixs_dic[key] = mtx

