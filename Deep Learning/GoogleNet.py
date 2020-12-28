import os
import cv2
import pandas as pd 
import numpy as np
import hyperspy.api as hs
import tensorflow as tf
import tensorflow_addons as tfa
from keras import optimizers,regularizers
from keras import backend as K
import tensorflow.keras as keras
from functions import escape_square_brackets
from computervision import erode,clahe,matnorm255,dm3toimg
K.clear_session()
from tensorflow.keras.models  import Sequential, load_model, model_from_json ,Model
from keras.utils import Sequence, multi_gpu_model, to_categorical
from keras.layers import concatenate, Input,Conv2D,MaxPooling2D,Dense,Activation,Flatten,GlobalAveragePooling2D,LSTM,Dropout,AveragePooling2D,BatchNormalization,GRU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
save_path = '/home/imeunu96/ML/googlenet/'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
generate = True
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

ref = get_reference(100)

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

callbacks_list  = [
    EarlyStopping( monitor = 'loss',  min_delta=0.001, patience=100,
                  verbose=1, mode='auto'),
    ModelCheckpoint(filepath = save_path+"/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5",
                    monitor = 'val_loss',
                    save_best_only=True ),
    ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7, verbose=1, min_delta=1e-4)]

# Inception 모듈 정의
def inception_module(x, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32):
    """
    # Arguments 
    x : 입력
    o_1 : 1x1 convolution 연산 출력값의 채널 수 (filter의 갯수)
    r_3 : 3x3 convolution 이전에 있는 1x1 convolution의 출력값 채널 수
    o_3 : 3x3 convolution 연산 출력값의 채널 수 

    r_5 : 5x5 convolution 이전에 있는 1x1 convolution의 출력값 채널 수 
    o_5 : 5x5 convolution 연산 출력값의 채널 수 

    pool: Averagepooling 다음의 1x1 convolution의 출력값 채널 수
    
    # returns
    4 종류의 연산의 결과 값을 채널 방향으로 합친 결과 
    """
    
    x_1 = Conv2D(filters = o_1,kernel_size= 1, padding='same', activation='relu')(x)
    
    x_2 = Conv2D(r_3, 1, padding='same', activation='relu')(x)
    x_2 = Conv2D(o_3, 3, padding='same', activation='relu')(x_2)
    
    x_3 = Conv2D(r_5, 1, padding='same', activation='relu')(x)
    x_3 = Conv2D(o_5, 5, padding='same', activation='relu')(x_3)
    
    x_4 = AveragePooling2D(pool_size=3, strides=1, padding='same')(x)
    x_4 = Conv2D(pool, 1, padding='same', activation='relu')(x_4)
    
    return concatenate([x_1, x_2, x_3, x_4])

def googlenet_model():
  input_shape = (100,100,1)
  input_ = Input(shape=input_shape)


  conv_1 = Conv2D(64, 7, padding='same', activation='relu')(input_)
  pool_1 = AveragePooling2D(pool_size=2, padding='same')(conv_1)
  LRN_1  = BatchNormalization()(pool_1)

  conv_2 = Conv2D(64, 1,  padding='valid', activation='relu')(LRN_1 )
  conv_3 = Conv2D(192, 3, padding='same', activation='relu')(conv_2 )
  LRN_2  = BatchNormalization()(conv_3)
  pool_2 = AveragePooling2D(pool_size=2, padding='same')(LRN_2)

  inception_3a = inception_module(pool_2, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32)
  inception_3b = inception_module(inception_3a, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
  pool_3 = AveragePooling2D(pool_size=3, strides=2, padding='same')(inception_3b )

  inception_4a = inception_module(pool_3 , o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)
  inception_4b = inception_module(inception_4a , o_1=160, r_3=112, o_3=224, r_5=24, o_5=64, pool=64)
  inception_4c = inception_module(inception_4b , o_1=128, r_3=128, o_3=256, r_5=24, o_5=64, pool=64)
  inception_4d = inception_module(inception_4c , o_1=112, r_3=144, o_3=288, r_5=32, o_5=64, pool=64)
  inception_4e = inception_module(inception_4d , o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
  pool_4 = AveragePooling2D(pool_size=3, strides=2, padding='same')(inception_4e )
  
  inception_5a  = inception_module(pool_4, o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
  inception_5b  = inception_module(inception_5a , o_1=384, r_3=192, o_3=384, r_5=48, o_5=128, pool=128)
  avg_pool  = GlobalAveragePooling2D()(inception_5b )
  dropout  = Dropout(0.4)(avg_pool)
  linear  = Dense(50, activation='relu')(dropout)
  linear2 = Dense(20, activation='relu')(linear)
  output = Dense(2, activation='softmax', name='main_classifier')(linear2)
  

  # Auxiliary Classifier
  auxiliary_4a = AveragePooling2D(5, strides=3, padding='valid')(inception_4a)
  auxiliary_4a = Conv2D(128, 1, strides=1, padding='same', activation='relu')(auxiliary_4a)
  auxiliary_4a = Flatten()(auxiliary_4a)
  auxiliary_4a = Dense(50, activation='relu')(auxiliary_4a)
  auxiliary_4a = Dropout(0.3)(auxiliary_4a)
  auxiliary_4a = Dense(20, activation='relu')(auxiliary_4a)
  auxiliary_4a = Dense(2, activation='softmax', name='auxiliary_4a')(auxiliary_4a)
  
  auxiliary_4d = AveragePooling2D( 5, strides=3, padding='valid')(inception_4d)
  auxiliary_4d = Conv2D(128,  1, strides=1, padding='same', activation='relu')(auxiliary_4d)
  auxiliary_4d = Flatten()(auxiliary_4d)
  auxiliary_4d = Dense(50, activation='relu')(auxiliary_4d)
  auxiliary_4d = Dropout(0.3)(auxiliary_4d)
  auxiliary_4d = Dense(20, activation='relu')(auxiliary_4d)
  auxiliary_4d = Dense(2, activation='softmax', name='auxiliary_4d')(auxiliary_4d) 

  #googlenet = Model(input_, [output, auxiliary_4a, auxiliary_4d])
  googlenet = Model(input_, output)
  return googlenet

def generate_model(summary=True):
    model = googlenet_model()
    if summary:
        model.summary()
    
    adam = optimizers.Adam(lr=0.00001 , clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def save_history_plot(hist,path):
    import matplotlib.pyplot as plt
    plt.ioff()
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    if 'auxiliary_4a_loss' in hist.history:
        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        acc_ax.plot(hist.history['main_classifier_accuracy'], 'b', label='accuracy')
        acc_ax.plot(hist.history['val_main_classifier_accuracy'], 'g', label='val accuracy')
    else:
        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        acc_ax.plot(hist.history['accuracy'], 'b', label='accuracy')
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
        if 'top_k_categorical_accuracy' in hist.history:
            acc_ax.plot(hist.history['val_top_k_categorical_accuracy'], label='val top_k_accuracy')
            acc_ax.plot(hist.history['top_k_categorical_accuracy'], label='top_k_accuracy')
    
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')  
    loss_ax.legend(loc='upper left')  
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper right')
    plt.savefig(path)
    # plt.show()
    plt.close('all')

if generate:
    model = generate_model()
    history =model.fit(train_X, train_Y, epochs=200, batch_size=512, validation_data=(test_X,test_Y), callbacks = callbacks_list)
else:    #load model
    json_file = open("/home/imeunu96/ML/googlenet/model.json", "r") 
    loaded_model_json = json_file.read() 
    json_file.close() 
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("/home/imeunu96/ML/googlenet/model_weight.h5") 
    print("Loaded model from disk")
    adam = optimizers.Adam(lr=0.00001 , clipvalue=0.5)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history =loaded_model.fit(train_X, train_Y, epochs=200, batch_size=512, validation_data=(test_X,test_Y), callbacks = callbacks_list)

#Model Save

save_history_plot(history, save_path+"/model_history_img.png")
pd.DataFrame(history.history).to_csv(save_path+"/history.csv")

if generate:
    try:
        print("-- Evaluate --")
        #scores = model.evaluate_generator(my_test_batch_generator ,  workers = workers, use_multiprocessing = True)
        scores = model.evaluate(test_X, test_Y, batch_size=512)
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
        '''
        answer_df = pd.DataFrame({'id':dataset.test_list, 'answer':answer_classes, 'predict':predicted_classes})
        concat_df = pd.concat([answer_df,predict_df],axis=1)
        concat_df.to_csv(save_path+"/test_answer.csv")
        '''
        acc = 0
        acc = accuracy_score(answer_classes, predicted_classes)
        print('test accuracy :', acc)
    except:
        pass
    # model save
    model_json = model.to_json()
    with open(save_path+"/model_acc_{:.4f}.json".format(acc), 'w') as json_file:
        json_file.write(model_json)

    # weight save
    model.save_weights(save_path +"/model_weight.h5")
    model_json = model.to_json()
    with open(save_path+"/model.json".format(acc), 'w') as json_file:
        json_file.write(model_json)
else:
    try:
        print("-- Evaluate --")
        #scores = model.evaluate_generator(my_test_batch_generator ,  workers = workers, use_multiprocessing = True)
        scores = loaded_model.evaluate(test_X, test_Y, batch_size=512)
        print("%s: %.2f%%" %(loaded_model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" %(loaded_model.metrics_names[1], scores[1]*100))

        print("-- Predict --")
        #output = model.predict_generator(my_test_batch_generator, workers = workers, use_multiprocessing = True)
        output = loaded_model.predict(test_X)
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
        '''
        answer_df = pd.DataFrame({'id':dataset.test_list, 'answer':answer_classes, 'predict':predicted_classes})
        concat_df = pd.concat([answer_df,predict_df],axis=1)
        concat_df.to_csv(save_path+"/test_answer.csv")
        '''
        acc = 0
        acc = accuracy_score(answer_classes, predicted_classes)
        print('test accuracy :', acc)
    except:
        pass
    
    # model save
    model_json = loaded_model.to_json()
    with open(save_path+"/model_googlenet.json", 'w') as json_file:
        json_file.write(model_json)

    # weight save
    loaded_model.save_weights(save_path +"/model_weight.h5")
    print("weight saved, train successful")
