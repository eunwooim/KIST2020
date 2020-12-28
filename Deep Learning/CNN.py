import os
import re
import cv2
import csv
import warnings
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
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dense,Activation,Flatten,GlobalAveragePooling2D,GlobalMaxPooling2D,Dropout,AveragePooling2D,BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.metrics import accuracy_score
from tensorflow.python.client import device_lib
from computervision import matnorm255,clahe,erode,dm3toimg
from functions import escape_square_brackets
warnings.filterwarnings('ignore')
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
save_path = '/home/imeunu96/ML/code/savehere/'

#data load
def save_history_plot(hist,path):
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

noise = np.random.normal(-5e-4,5e-4,(100,100))
ref = matnorm255(hs.load('/home/imeunu96/diffraction_patterns/Dataset2_Hour_00_Minute_00_Second_06_Frame_0001.dm3').data)
ref = clahe(ref.astype(np.uint8),cliplimit=5)
ref = ref + noise
ref = erode(ref)
ref = matnorm255(ref)

#Generating Dataset
docX = []
docY = []
f = pd.read_excel('/home/imeunu96/ML/code/savehere/lesseph.xlsx')
df = f.copy()
indexNames = df[ df['classified'] == 2 ].index
df.drop(indexNames , inplace=True)
df.to_excel('/home/imeunu96/ML/code/savehere/lesseph_without2.xlsx',index=False)

f = pd.read_excel('/home/imeunu96/ML/code/savehere/lesseph_without2.xlsx')
for classified in f['classified']:
    for i in range(8):
        docY.append(classified)
print('Y_data Loaded')
# filename = np.array(x)
Y_data = np.array(docY)
for filepath in f['filepath']:
    try:
        print('loading %s' % (filepath))
        path = escape_square_brackets(filepath)
        img = dm3toimg(path,ref).astype(np.uint8)
        for i in range(0,360,45):
            (h, w) = img.shape[:2]
            (cX, cY) = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D((cX, cY), i, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            docX.append(rotated)
    except:
        print('failed %s' %(filepath))
        nx = np.where(f['filepath'] == filepath)
        Y_data = np.delete(Y_data, nx[0][0])
X_data = np.array(docX)
docX.clear();docY.clear()
np.save('/home/imeunu96/ML/code/savehere/X_data.npy',X_data)
np.save('/home/imeunu96/ML/code/savehere/Y_data.npy',Y_data)

#Loading Dataset
X_data = np.load('/home/imeunu96/ML/1126_cnn8aug/X_data.npy')
Y_data = np.load('/home/imeunu96/ML/1126_cnn8aug/Y_data.npy')
X_data = np.expand_dims(X_data,axis = -1)
Y_data = np.expand_dims(Y_data,axis = -1)
print('X_data len : %d, Y_data len : %d' % (len(X_data), len(Y_data)))

#randomly choose 20% test data
tot_ix = range(len(Y_data))
test_ix = np.random.choice(tot_ix, int(len(Y_data)*0.2), replace=False)
test_ix = np.sort(test_ix,axis=0)
train_ix = list(set(tot_ix) - set(test_ix))

#write test data index into csv files
test_ix = np.reshape(test_ix, test_ix.shape + (1,))
mat1 = test_ix
dataframe1 = pd.DataFrame(data=mat1.astype(int))
dataframe1.to_csv('/home/imeunu96/ML/code/savehere/20pertestset.csv', sep=',', header=False,
                  float_format = '%.2f', index = False)

#load test index and convert to hot vector
test_index = pd.read_csv('/home/imeunu96/ML/code/savehere/20pertestset.csv', header=None)
test_ix = test_index[0]
tot_ix = range(len(Y_data))
train_ix = list(set(tot_ix) - set(test_ix))
test_X = X_data[test_ix]
train_X = X_data[train_ix]
test_Y = Y_data[test_ix]
train_Y = Y_data[train_ix]
from keras.utils.np_utils import to_categorical
class2 = ['ring','spot']
train_Y = to_categorical(train_Y, 2)
test_Y = to_categorical(test_Y, 2)

#shuffle data before training
tot_ix =range(len(train_X))
rand_ix = np.random.choice(tot_ix, len(train_X), replace=False)
train_X = train_X[rand_ix]
train_Y = train_Y[rand_ix]
print(train_X.shape)

#Model
def generate_model_cnn(mode='patterns', pooling=0, layers=3, kernel=11):
    model = Sequential()
    print("=================================CNN=====================================")
    model.add(Conv2D(filters=100, kernel_size=(15,15), strides = 1, padding ='same', input_shape=(100,100,1), activation = 'relu')) #add convolution layer
    model.add(Dropout(0.2))
    model.add(AveragePooling2D(pool_size=(3,3), strides=2)) #pooling layer
    model.add(Conv2D(filters=100, kernel_size=(20,20), strides = 3, padding ='same', activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(AveragePooling2D(pool_size=(3,3), strides=None))
    model.add(Conv2D(filters=80, kernel_size=(25,25), strides = 2, padding ='same', activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(AveragePooling2D(pool_size=(2,2), strides=None))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    model.add(Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    return model

def generate_model(mode='patterns', summary=True):
    model = generate_model_cnn(mode=mode, pooling=0)
    if summary:
        model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

callbacks_list  = [
    EarlyStopping( monitor = 'loss',  min_delta=0.001, patience=100,
                  verbose=1, mode='auto'),
    ModelCheckpoint(filepath = save_path+"/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5",
                    monitor = 'val_loss',
                    save_best_only=True ),
    ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7, verbose=1, min_delta=1e-4)]

model = generate_model(mode='patterns',summary=True)
history = model.fit(train_X, train_Y, epochs=300, batch_size=100, validation_data=(test_X,test_Y),callbacks=callbacks_list)

save_history_plot(history, save_path+"/model_history_img.png")
pd.DataFrame(history.history).to_csv(save_path+"/history.csv")

print("-- Evaluate --")
scores = model.evaluate(test_X, test_Y, batch_size=100)
print("%s: %.2f%%" %(model.metrics_names[0], scores[0]))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
#output = model.predict_generator(my_test_batch_generator, workers = workers, use_multiprocessing = True)
output = model.predict(test_X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

tt=[]
for i in range(len(output)):
    temp=[]
    temp.append(output[i].argsort()[::-1][:5] )
    temp.append(np.round(np.sort(output[i])[::-1][:5],6))
    tt.append(temp)
predict_df = pd.DataFrame(tt)

predicted_classes= output.argmax(axis=1)
answer_classes= test_Y.argmax(axis=1)
acc = accuracy_score(answer_classes, predicted_classes)
print('test accuracy :', acc)

# model save
model_json = model.to_json()
with open(save_path+"/model_acc_{:.4f}.json".format(acc), 'w') as json_file:
    json_file.write(model_json)

# weight save
model.save_weights(save_path +"/model_weight.h5")
model_json = model.to_json()
with open(save_path+"/model.json".format(acc), 'w') as json_file:
    json_file.write(model_json)
