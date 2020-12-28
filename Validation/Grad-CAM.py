import os
import re
import cv2
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
tf.compat.v1.disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"] = ''
save_path = '/home/imeunu96/ML/code/savehere/'
from computervision import matnorm255

def load_model(jsonpath,weightpath):
    '''Load Model with Architecture(.json) and Weight(.hdf5)'''
    with open(jsonpath) as f:
        json = f.read()
    model = tf.keras.models.model_from_json(json)
    model.load_weights(weightpath)
    return model

def get_bgr_tensor(img,size=100):
    '''Convert Gray Image to BGR Image'''
    img = img.astype(np.uint8)
    img = cv2.resize(img,dsize=(size,size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    return rgb

def get_gradcam_heatmap(img,model,layername='conv2d'):
    '''Obtain Grad CAM from model'''
    class1_output = model.output[:, 1]
    last_conv_layer = model.get_layer(layername) 
    grads = K.gradients(class1_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(4):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    gradCAM = np.maximum(heatmap, 0)
    gradCAM /= np.max(gradCAM)
    return heatmap

def heatmap2bgr(heatmap):
    '''Normalize and return as BGR image'''
    heatmap = matnorm255(heatmap,100)
    heatmap = heatmap.astype(np.uint8)
    colormap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
    return colormap

def get_superimposed(img1,img2,ratio=0.6):
    '''
    img1 : heatmap
    img2 : image
    ratio : RealNumber in [0,1], higher img1 clearer
    '''
    import matplotlib.cm as cm
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:,:3]
    boolmap = matnorm255(heatmap,100)
    boolmap = boolmap.astype(np.uint8)
    jet_heatmap = jet_colors[boolmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((100,100))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap*0.6+bgr
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

#Load Dataset
X_data = np.load('/home/imeunu96/ML/1130_GoogleNet/X_data.npy')
Y_data = np.load('/home/imeunu96/ML/1130_GoogleNet/Y_data.npy')
X_data = np.expand_dims(X_data,axis = -1)
Y_data = np.expand_dims(Y_data,axis = -1)
#X_data = np.load('/home/imeunu96/ML/uncertain.npy')
#X_data = np.expand_dims(X_data,axis = -1)
print('Dataset loaded')

#Load Models
models = ['1130_GoogleNet/','1126_cnn8aug/']
ringpath = 'ringCAM'
spotpath = 'spotCAM'
for modelname in models:
    count=0
    jsonpath = '/home/imeunu96/ML/'+modelname+'model.json' 
    weightpath='/home/imeunu96/ML/'+modelname+'best.hdf5'
    model = load_model(jsonpath,weightpath)
    print('Model Loaded', model)
    for i in range(len(X_data)):
        img = X_data[i]                        #2d array
        clss = Y_data[i]                       #class
        gray = np.expand_dims(img,axis=-1)     #3d gray
        bgr = get_bgr_tensor(gray)             #3d bgr
        tensor = np.expand_dims(img,axis=0)    #4d array
        heatmap = get_gradcam_heatmap(tensor,model)
        colormap = heatmap2bgr(heatmap)
        CAM = np.array(get_superimposed(colormap,bgr))
        imgpath = ringpath if clss==0 else spotpath
        os.chdir('/home/imeunu96/ML/code/savehere/'+modelname+imgpath)
        plt.imsave(str(count)+'.png',CAM)
        print(clss, count)
        count+=1
