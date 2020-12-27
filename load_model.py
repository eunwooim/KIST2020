import os
import re
import cv2
import csv
import numpy as np
import pandas as pd 
import tensorflow as tf
from shutil import copy2
import hyperspy.api as hs
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model, model_from_json 
from tensorflow.keras.utils import Sequence, multi_gpu_model, to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,Dense, Activation, Flatten,GlobalAveragePooling2D, LSTM,GlobalMaxPooling2D,Dropout,AveragePooling2D,BatchNormalization, GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.metrics import accuracy_score
from tensorflow.python.client import device_lib
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
save_path = '/home/imeunu96/ML/code/savehere/'

##########################################data generate##########################################

#data load

def save_history_plot(hist,path):
    import matplotlib.pyplot as plt
    plt.ioff()
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    acc_ax.plot(hist.history['accuracy'], 'b', label='accuracy')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
    if 'top_k_categorical_accuracy' in hist.history:
        acc_ax.plot(hist.history['val_top_k_categorical_accuracy'], label='val top_k_accuracy')
        acc_ax.plot(hist.history['top_k_categorical_accuracy'], label='top_k_accuracy')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper right')
    plt.savefig(path)
    # plt.show()
    plt.close('all')

def matnorm255(matrix,size=100):
    '''Convert raw matrix into uint8 image matrix with Adjusted Contrast'''
    x = abs(matrix)
    v = x.max()
    x[x>v]=v
    gray = 255*(x-x.min())/(x.max()-x.min())
    gray = gray.astype(np.float32)
    gray = cv2.resize(gray,dsize=(size, size), interpolation=cv2.INTER_AREA)
    return gray

def clahe(img,cliplimit=2):
    '''Apply Contrast Limited Adaptive Histogram Equalization'''
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8,8))
    claheimg = clahe.apply(img)
    return claheimg

def erode(img,kersize=3,iteration=2):
    '''Morphology Erosion'''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kersize,kersize))
    erode = cv2.erode(img, kernel, anchor=(-1, -1), iterations=iteration)
    return erode

def dm3toimg(filepath,ref):
    x = hs.load(filepath).data
    img = matnorm255(x,len(ref))
    img = match_histograms(img,ref,multichannel=False)
    return img

def escape_square_brackets(text):
    rep = dict((re.escape(k), v) for k, v in {"[": "[[]", "]": "[]]"}.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

def get_reference(size):
    noise = np.random.normal(-5e-4,5e-4,(size,size))
    ref = matnorm255(hs.load('/home/imeunu96/diffraction_patterns/Dataset2_Hour_00_Minute_00_Second_06_Frame_0001.dm3').data,size)
    ref = clahe(ref.astype(np.uint8),cliplimit=5)
    ref = ref + noise
    ref = erode(ref)
    ref = matnorm255(ref,size)
    return ref
ref1 = get_reference(100)
ref5 = get_reference(500)

def load_model(jsonpath,weightpath):
    with open(jsonpath) as f:
        json = f.read()
    model = tf.keras.models.model_from_json(json)
    model.load_weights(weightpath)
    return model

def save_big_img(imgpath,savename):
    path = escape_square_brackets(imgpath)
    ref = get_reference(600)
    img = dm3toimg(path,ref)
    plt.imsave(savename,img)
    return

def rotate_img(img,angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

#Load Dataset
class2 = ['ring','spot']
X_data = np.load("/home/imeunu96/ML/1129_8dense/X_data.npy")
print(len(X_data))
Y_data = np.load("/home/imeunu96/ML/1129_8dense/Y_data.npy")
X_data = np.expand_dims(X_data,axis = -1)
Y_data = np.expand_dims(Y_data,axis = -1)
test_index = pd.read_csv('/home/imeunu96/ML/1130_GoogleNet/20pertestset.csv', header=None)
test_ix = test_index[0]
tot_ix = range(len(Y_data))
train_ix = list(set(tot_ix) - set(test_ix))
test_X = X_data[test_ix]
train_X = X_data[train_ix]
test_Y = Y_data[test_ix]
train_Y = Y_data[train_ix]
print('====Dataset Loaded====')

#Load Model
jsonpath = '/home/imeunu96/ML/1130_GoogleNet/model.json'
weightpath = '/home/imeunu96/ML/1130_GoogleNet/best.hdf5'
load_model(jsonpath,weightpath)
print('====Model Loaded====')

#Predicting Test set
y_predict = model.predict(test_X)
count=0;err=[]
for data in y_predict:
    if data[0]>data[1]:
        y_pred = 0
        if y_pred != test_Y[count]:
            err.append(count)
    else:
        y_pred = 1
        if y_pred != test_Y[count]:
            err.append(count)
    count += 1
print('=====Test set Result=====')
print('Ring Success',rs)
print('Ring Fail',rf)
print('Spot Success',ss)
print('Spot Fail',sf)
'''
##Plot test set errors
mlpath = '/home/imeunu96/ML/'
paths = ['1129_8dense']

for path in paths:
    jsonpath = '/home/imeunu96/ML/'+path+'/model.json' 
    with open(jsonpath) as f:
        json = f.read()
    model = tf.keras.models.model_from_json(json)
    weightpath='/home/imeunu96/ML/'+path+'/best.hdf5'
    model.load_weights(weightpath)
    X_data = np.load('/home/imeunu96/ML/'+path+'/X_data.npy')
    Y_data = np.load('/home/imeunu96/ML/'+path+'/Y_data.npy')
    X_data = np.expand_dims(X_data,axis = -1)
    Y_data = np.expand_dims(Y_data,axis = -1)
    test_index = pd.read_csv('/home/imeunu96/ML/'+path+'/20pertestset.csv', header=None)
    path_index = pd.read_excel('/home/imeunu96/ML/code/8aug.xlsx')
    test_ix = test_index[0]
    tot_ix = range(len(Y_data))
    train_ix = list(set(tot_ix) - set(test_ix))
    test_X = X_data[test_ix]
    train_X = X_data[train_ix]
    test_Y = Y_data[test_ix]
    train_Y = Y_data[train_ix]
    y_predict = model.predict(test_X)
    count=0;rs=0;rf=0;ss=0;sf=0;err=[]
    for data in y_predict:
        if data[0]>data[1]:
            y_pred = 0
            if y_pred == test_Y[count]:
                rs+=1
            else:
                rf+=1
                err.append(count)
                p=path_index['filepath'][count]
                save_big_img(p,save_path+path+str(count)+'-'+str(data[0])+'-s.png')
        else:
            y_pred = 1
            if y_pred == test_Y[count]:
                ss+=1
            else:
                sf+=1
                err.append(count)
                p=path_index['filepath'][count]
                save_big_img(p,save_path+path+str(count)+'-'+str(data[1])+'-r.png')
        count += 1
    print('====='+path+' Test set Result=====')
    print('Ring Success',rs)
    print('Ring Fail',rf)
    print('Spot Success',ss)
    print('Spot Fail',sf)
    print('accuracy : ',(rs+ss)/(rs+rf+ss+sf))
'''
#Generate uncertain(2) array
f = pd.read_excel('/home/imeunu96/ML/code/lesseph.xlsx')
df = f.copy();count=0;docX=[]
imgpath = '/home/imeunu96/ML/code/imgpath/'
indexNames = pd.Series(df[ df['classified'] == 2 ].index)
paths=np.array(df.drop(columns='classified'))
plt.figure(figsize=(5,5))
plt.tight_layout()
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
plt.axis('off'), plt.xticks([]), plt.yticks([])
for filepath in paths[indexNames]:
    path = escape_square_brackets(filepath[0])
    img = dm3toimg(path,ref1).astype(np.uint8)
    docX.append(img)
    # save = dm3toimg(path,ref5).astype(np.uint8)
    # plt.imsave(imgpath+str(count)+'.png',save)
    count+=1
Xdata = np.array(docX);docX.clear();count=0
np.save('/home/imeunu96/ML/code/savehere/uncertain.npy',Xdata)
print('====Image array Saved====')
'''
#Load and predict uncertain(2) array
Xdata = np.load('/home/imeunu96/ML/uncertain.npy')
Xdata = np.expand_dims(Xdata,axis = -1)
folders = ['1130_GoogleNet']
for folder in folders:
    count=0
    d = '/home/imeunu96/ML/'+folder+'/'
    os.chdir(d)
    jsonpath = d+'model.json'
    weightpath = d+'best.hdf5'
    model = load_model(jsonpath,weightpath)
    y_predict = model.predict(Xdata)
    f = open(d+'/predict_uncertain.csv','w')
    for data in y_predict:
        if data[0]>data[1]:
            f.write(str(count)+',ring,'+str(data[0])+'\n')
        else:
            f.write(str(count)+',spot,'+str(data[1])+'\n')
        count+=1
    f.close()
print('====Prediction Saved====')
