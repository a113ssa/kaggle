# %% [code]

# paste the folowing code in notebook to include this utils
# import os
# import sys 
# sys.path.append(os.path.abspath('/kaggle/usr/lib/utils/utils.py'))
# import utils

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

def plot_train_results(metrics, epochs, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    fix, ax = plt.subplots(figsize=(15, 4))
    
    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name, ]
        
    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])
        
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, epochs])
    plt.ylim(ylim)
    
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()

def plot_loss_and_accuracy(history):
    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    
    plot_train_results(
        [train_loss, val_loss],
        ylabel='Loss',
        ylim=[0.0, 5.0],
        metric_name=['Training Loss', 'Validation Loss'],
        color=['orange', 'g']
    )
    
    plot_train_results(
        [train_acc, val_acc],
        ylabel='Accuracy',
        ylim=[0.0, 1.0],
        metric_name=['Training Accuracy', 'Validation Accuracy'],
        color=['orange', 'g']
    )


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