import os
import numpy as np
import tensorflow as tf
from keras import optimizers,regularizers
from keras import backend as K
from keras.models import Sequential, load_model, model_from_json 
from keras.utils import Sequence, multi_gpu_model, to_categorical
from keras.layers import Input, Conv1D, MaxPooling1D,Dense, Activation, Flatten,GlobalAveragePooling1D, LSTM,GlobalMaxPooling1D,Dropout,AveragePooling1D,BatchNormalization, GRU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# callbacks_list  = [
#     EarlyStopping( monitor = 'loss',  min_delta=0.001, patience=10,
#                   verbose=1, mode='auto'),

#     ModelCheckpoint(filepath = save_path+"/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5",
#                     monitor = 'val_loss',
#                     save_best_only=True ),

#     # 변화 없으면 학습속도를 줄이자
#     ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7, verbose=1, min_delta=1e-4)
# ]

class My_Custom_Generator(Sequence) :
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size):
        'Initialization'
        self.list_IDs = list_IDs        # image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
                #에포크당 배치 수 만큼 나타냄
            # 생성기가 생성할 배치 수 계산 -> 총 샘플 / 배치 = 해당 값 반환
        return (np.ceil(len(self.list_IDs) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx): 
        'Generate one batch of data'
        batch_x = self.list_IDs[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        return get_dataset(batch_x), np.array(batch_y)

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

def generate_model_cnn(mode='spacegroup', pooling=0, layers=3, kernel=11):
  model = Sequential()
  model.add(Conv1D(filters=30, kernel_size=kernel, activation='relu', input_shape=(512,1)))
  model.add(MaxPooling1D(pool_size=2))
  # model.add(AveragePooling1D(pool_size=2))
  layers -= 1
  layer = 1 # for layers = 0
  for layer in range(layers):
    model.add(Conv1D(filters=50, kernel_size=kernel, activation='relu'))
    if layer%2==0:
      model.add(MaxPooling1D(pool_size=2))
  if layer%2==1:
    model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())

  if mode=='crystalsystems':
    model.add(Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
  elif mode=='bravais':
    model.add(Dense(3000, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(14, activation='softmax'))
  else:
    target_dim = 230
    current_dim = np.int64(model.layers[-1].output_shape[1]/2)
    dropout_rate = 0.2
    while(True):
      if current_dim <= target_dim*2:
        break
      model.add(Dense(current_dim, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
      model.add(Dropout(dropout_rate))
      current_dim = np.int64(current_dim/2)
      dropout_rate += 0.2
      if dropout_rate > 0.5:
        dropout_rate = 0.5
    model.add(Dense(target_dim, activation='softmax'))
  return model

def generate_model_paper(mode='spacegroup',pooling=0):
  if mode=='spacegroup':  
    model = Sequential()
    # init_func = 'he_uniform' #,kernel_initializer=init_func
    model.add(Conv1D(filters=2, kernel_size=9, input_shape=(512,1)  ))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D())

    model.add(Conv1D(filters=2, kernel_size=7))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D())

    model.add(Conv1D(filters=4, kernel_size=7))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D())

    model.add(Conv1D(filters=8, kernel_size=5))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D())

    model.add(Conv1D(filters=12, kernel_size=3))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D())

    model.add(Dropout(0.10))
    model.add(Conv1D(filters=230, kernel_size = 1))
    # model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    # model.add(Dense(230))
    model.add(Activation('softmax'))
  elif mode=='crystalsystems':
    model = Sequential()
    model.add(Conv1D(filters=2, kernel_size=151, input_shape=(512,1)  ))
    model.add(Activation('relu'))
    model.add(AveragePooling1D())
    model.add(Conv1D(filters=4, kernel_size=71))
    model.add(Activation('relu'))
    model.add(AveragePooling1D())
    model.add(Conv1D(filters=8, kernel_size=31))
    model.add(Activation('relu'))
    model.add(AveragePooling1D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(7, activation='softmax'))
  return model

def generate_model_a_cnn(mode='spacegroup',pooling=0):
  if mode=='spacegroup':  
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=11, activation='relu', input_shape=(512,1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=20, kernel_size=11, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=230, kernel_size=11, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(230, activation='softmax'))
  return model #10분 -> 62...0.27

def generate_model_test(mode='spacegroup',pooling=0):
  if mode=='spacegroup':

    model = Sequential()
    # init_func = 'he_uniform' 
    model.add(Conv1D(filters=20, kernel_size=90, input_shape=(512,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=20, kernel_size=70))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=40, kernel_size=70))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=50, kernel_size=50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(6000,activity_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(2500,activity_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(1200,activity_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(600,activity_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(230))
    model.add(Activation('softmax'))
  return model

def generate_model_lstm(mode='spacegroup'):
  model = Sequential()
  model.add(LSTM(256, input_shape=(4096,1)))
  if mode=='crystalsystems':
    model.add(Dense(7, activation='softmax'))
  elif mode=='bravais':
    model.add(Dense(14, activation='softmax'))
  else:
    model.add(Dense(230, activation='softmax'))
  return model

