# %% [code]
from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url, save_path):
    print('Downloading and extracting assets...', end='')
    
    urlretrieve(url, save_path)
    
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
            
        print('Done')
        
    except Exception as e:
        print("\nInvalid file.", e)


import numpy as np

def shuffle_and_split_data(data, test_ratio, seed = None):
    if (seed != None):
        np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]