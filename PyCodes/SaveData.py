import pickle

def save_to_txt(data, filename):
    ''' Function to Save variables to TXT files '''
    file_open = open(filename, 'wb')
    pickle.dump(data, file_open)
    file_open.close()
    return True

def load_from_txt(filename):
    ''' Function to Load variables from TXT files '''
    file_open = open(filename, 'rb')
    data = pickle.load(file_open)
    file_open.close()
    return data
