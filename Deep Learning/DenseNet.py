import os
import re
import cv2
import csv
import numpy as np
import pandas as pd 
import hyperspy.api as hs
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from skimage.exposure import match_histograms
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras import backend as K
from keras.applications import DenseNet121
from tensorflow.keras.models import Sequential, load_model, model_from_json 
from tensorflow.keras.utils import Sequence, multi_gpu_model, to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,Dense, Activation,Flatten,GlobalAveragePooling2D, LSTM,Dropout,AveragePooling2D,BatchNormalization, GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.metrics import accuracy_score
from tensorflow.python.client import device_lib
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from functions import escape_square_brackets
from computervision import matnorm255,clahe,erode,dm3toimg
#model = keras.applications.densenet.DenseNet201()
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
save_path = '/home/imeunu96/ML/code/savehere/'

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

#Loading Dataset
X_data = np.load('/home/imeunu96/ML/code/savehere/X_data.npy')
Y_data = np.load('/home/imeunu96/ML/code/savehere/Y_data.npy')
X_data = np.expand_dims(X_data,axis = -1)
Y_data = np.expand_dims(Y_data,axis = -1)

#randomly choose 20% test data
tot_ix = range(len(Y_data))
test_ix = np.random.choice(tot_ix, int(len(Y_data)*0.2), replace=False)
test_ix = np.sort(test_ix,axis=0)
train_ix = list(set(tot_ix) - set(test_ix))

#write test data index into csv files
test_ix = np.reshape(test_ix, test_ix.shape + (1,))
mat1 = test_ix
dataframe1 = pd.DataFrame(data=mat1.astype(int))
dataframe1.to_csv('20pertestset.csv', sep=',', header=False,
                  float_format = '%.2f', index = False)

#load test index and convert to hot vector
test_index = pd.read_csv('20pertestset.csv', header=None)
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
print('train_X.shape')
print(train_X.shape)

#Model
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1
        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {_val_kappa:.4f}")      
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')
        return

def build_model():
    model = Sequential()
    model.add(DenseNet121(weights=None,include_top=False,input_shape=(100,100,1)))
    model.add(AveragePooling2D(pool_size=(3,3), strides=None))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model

callbacks_list  = [
    EarlyStopping( monitor = 'loss',  min_delta=0.001, patience=100,
                  verbose=1, mode='auto'),
    ModelCheckpoint(filepath = save_path+"/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5",
                    monitor = 'val_loss',
                    save_best_only=True ),
    ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7, verbose=1, min_delta=1e-4)]

model = build_model()
model.summary()
kappa_metrics = Metrics()
history = model.fit(train_X,train_Y,epochs=200,batch_size=32,validation_data=(test_X,test_Y),callbacks=callbacks_list)
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
    temp.append(output[i].argsort()[::-1][:5])
    temp.append(np.round(np.sort(output[i])[::-1][:5],6))
    tt.append(temp)
predict_df = pd.DataFrame(tt)

predicted_classes = output.argmax(axis=1)
answer_classes = test_Y.argmax(axis=1)
acc = accuracy_score(answer_classes, predicted_classes)
print('test accuracy :', acc)

#Model save
model_json = model.to_json()
with open(save_path+"/model_acc_{:.4f}.json".format(acc), 'w') as json_file:
    json_file.write(model_json)

#Weight save
model.save_weights(save_path +"/model_weight.h5")
model_json = model.to_json()
with open(save_path+"/model.json".format(acc), 'w') as json_file:
    json_file.write(model_json)
