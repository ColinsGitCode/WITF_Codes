import random

def Drafts_samples(length,ratios):
    lis = [ ]
    for i in range(length):
        lis.append(i)
    sample_count = int(length * ratios)
    lis_sample = random.sample(lis,sample_count)
    return lis_sample

def Drafts_get_itemPos_itemIDs_dic(arr):
    dic = {}
    for pos in range(len(arr)):
        dic[pos] = arr[pos]

    return dic