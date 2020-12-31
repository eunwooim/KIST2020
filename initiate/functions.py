import os
import re
import cv2
import csv
import shutil
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import hyperspy.api as hs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from skimage.exposure import match_histograms

def walk(folder):
    """Walk through every files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename

def kth(arr, K):
    '''Find Kth largest number in array'''
    ORDINAL_MSG = ('st', 'nd', 'rd')[K-1] if K <= 3 else 'th'
    unique_set = set(arr)
    if len(unique_set) < K:
        raise IndexError(f"There's no {K}{ORDINAL_MSG} value in array")
    elif K <= 0 or not arr:
        raise ValueError("K should be over 0 and arr should not be empty")
    INF = float('inf')
    memory = [-INF] * K
    for n in arr:
        if n <= memory[-1]:
            continue
        for i, m in enumerate(memory):
            if (i == 0 and n > m) or m < n < memory[i-1]:
                for j in range(len(memory)-2, i-1, -1):
                    memory[j+1] = memory[j]
                memory[i] = n
                break
    return memory[-1]

def isdm3(ext):
    dmlist = ['.dm3', '.dm4']
    if ext in dmlist:
        return True
    else:
        return False

def loaddm3(filename):
    '''Load dm3 or dm4 raw matrix'''

    if name[-4:] in ['.dm3', '.dm4']:
        matrix = hs.load(filename.data)
    else:
        try:
            newname = filename+'.dm3'
            matrix = hs.load(newname).data
        except:
            newname = filename+'.dm4'
            matrix = hs.load(newname).data
    return matrix
  
def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
    plt.title("Confusion Matrix")
    sns.set(font_scale=1.2)
    ax = sns.heatmap(data, annot=True, cmap="Blues", cbar_kws={'label': 'Scale'},fmt='.4g')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="Actual", xlabel="Predicted")
    plt.show(output_filename)
    plt.close()
    plt.savefig(output_filename)
    # define data
    import numpy as np
    cm = np.array([[1979,3],
                [7,996]])
    # cm = np.array(cm)
    # define labels
    labels = labels
    # create confusion matrix
    plot_confusion_matrix(cm, labels, "C:/Users/im/Desktop/confusion_matrix.png")
    print('Accuracy : ',cm.trace()/cm.sum())

def plot_history(csvpath):
    '''Plot loss, accuracy of training, validation set'''
    loss=[];accuracy=[];val_loss=[];val_accuracy=[]
    with open(csvpath,'r') as f:
        rdf = csv.reader(f)
        for num in rdf:
            loss.append(num[1])
            accuracy.append(num[2])
            val_loss.append(num[3])
            val_accuracy.append(num[4])
    loss=np.array(loss[1:]).astype(np.float64)
    accuracy=np.array(accuracy[1:]).astype(np.float64)
    val_loss=np.array(val_loss[1:]).astype(np.float64)
    val_accuracy=np.array(val_accuracy[1:]).astype(np.float64)
    f.close()
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    plt.figure(figsize=(12,9))
    loss_ax.plot(loss, 'y', label='train loss')
    loss_ax.plot(val_loss, 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='lower right',bbox_to_anchor=(1.05, 1))
    acc_ax.plot(accuracy, 'b', label='train acc')
    acc_ax.plot(val_accuracy, 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper right',bbox_to_anchor=(1.3, 1.2))
