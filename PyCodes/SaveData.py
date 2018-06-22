# Python3 
# Filename : SaveData.py
# Usages : Save Data at local disks

import pickle

def SaveToTxt(Data, filename):
    f = open(filename, 'wb')
    pickle.dump(Data, filename)
    f.close()
    return True

def LoadFromTxt(filename):
    f = open(filename, 'rb')
    Data = pickle.load(f)
    f.close()
    return Data
    
