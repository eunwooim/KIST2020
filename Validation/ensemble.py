import os
import re
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import hyperspy.api as hs
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from skimage.exposure import match_histograms
from tensorflow.python.client import device_lib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.models import Sequential, load_model, model_from_json 
from tensorflow.keras.utils import Sequence, multi_gpu_model, to_categorical
tf.config.threading.set_inter_op_parallelism_threads(2)
save_path = '/home/imeunu96/ML/code/savehere/'
os.environ["CUDA_VISIBLE_DEVICES"] = ''
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
disable_eager_execution()

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

def save_big_img(imgpath,savename):
    path = escape_square_brackets(imgpath)
    ref = get_reference(600)
    img = dm3toimg(path,ref)
    plt.imsave(savename,img)
    return

def load_model(jsonpath,weightpath):
    with open(jsonpath) as f:
        json = f.read()
    model = tf.keras.models.model_from_json(json)
    model.load_weights(weightpath)
    return model

#Load Dataset
X_data = np.load('/home/imeunu96/ML/1126_cnn8aug/X_data.npy')
Y_data = np.load('/home/imeunu96/ML/1126_cnn8aug/Y_data.npy')
X_data = np.expand_dims(X_data,axis = -1)
Y_data = np.expand_dims(Y_data,axis = -1)
test_index = pd.read_csv('/home/imeunu96/ML/1126_cnn8aug/20pertestset.csv', header=None)
test_ix = test_index[0]
tot_ix = range(len(Y_data))
test_X = X_data[test_ix]
test_Y = Y_data[test_ix]
X_data = np.load('/home/imeunu96/ML/uncertain.npy')
X_data = np.expand_dims(X_data,axis = -1)
print(len(test_X), 'dataset loaded')

#Load Models
jsonpath = '/home/imeunu96/ML/1126_cnn8aug/model.json' 
weightpath='/home/imeunu96/ML/1126_cnn8aug/best.hdf5'
clf1 = load_model(jsonpath,weightpath)
clf1._estimator_type="classifier"
jsonpath = '/home/imeunu96/ML/1129_8dense/model.json' 
weightpath='/home/imeunu96/ML/1129_8dense/best.hdf5'
clf2 = load_model(jsonpath,weightpath)
clf2._estimator_type="classifier"
jsonpath = '/home/imeunu96/ML/1130_GoogleNet/model.json' 
weightpath='/home/imeunu96/ML/1130_GoogleNet/best.hdf5'
clf3 = load_model(jsonpath,weightpath)
clf3._estimator_type="classifier"
eclf = VotingClassifier(estimators = [('CNN',clf1), ('DenseNet',clf2), ('GoogleNet',clf3)], voting = 'soft')
total = clf1.predict(test_X)+clf2.predict(test_X)+clf3.predict(test_X)
pred = clf1.predict(X_data)+clf2.predict(X_data)+clf3.predict(X_data);pred=pred/3       #Ensemble Soft Voting Classifier
np.save('/home/imeunu96/ML/cnn_pred_uncertain.npy',clf1.predict(X_data))
np.save('/home/imeunu96/ML/dense_pred_uncertain.npy',clf2.predict(X_data))
np.save('/home/imeunu96/ML/google_pred_uncertain.npy',clf3.predict(X_data))
ensemble = np.expand_dims(np.argmax(total,axis=1),axis=-1)
count=0;err=[];p=[]
for prob in total:
    if prob[0]>prob[1]:
        y_pred = 0
        if y_pred != test_Y[count]:
            err.append(count);p.append(prob[0])
    else:
        y_pred = 1
        if y_pred != test_Y[count]:
            err.append(count);p.append(prob[0])
    count += 1
cm = confusion_matrix(test_Y, ensemble)
print(cm);print('Accuracy : ',cm.trace()/cm.sum())
print('Error : ',err)
print(p)

# for clf, label in zip([clf1, clf2, clf3, eclf], ['CNN', 'DenseNet', 'GoogleNet', 'Ensemble']):
#     scores = cross_val_score(clf, test_X, test_Y, scoring='accuracy', cv=5)
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# eclf.fit(train_X, train_Y)
# print('Ensemble Generated, Predicting ...')
# y_pred = eclf.predict(test_X)
# print(y_pred)
# print('Test accuracy:', accuracy_score(y_pred, test_Y))
