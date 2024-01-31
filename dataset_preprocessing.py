#!/usr/bin/env python
# coding: utf-8

# In[28]:


from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential,load_model
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from livelossplot.inputs.tf_keras import PlotLossesCallback
from keras import layers
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2

import itertools
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


image_size=(48,48)
def dataset_loader():
    df = pd.read_csv("fer2013.csv")
    width, height = 48, 48
    pics = []
    for pix in df["pixels"]:
        pic = [int(pixel) for pixel in pix.split(' ')]
        pic = np.asarray(pic).reshape(width, height)
        pic = cv2.resize(pic.astype('uint8'),image_size)
        pics.append(pic.astype('float32'))
    pics = np.asarray(pics)
    pics = np.expand_dims(pics, -1)
    emo = pd.get_dummies(df['emotion']).values
    return pics, emo,df


# In[3]:


def preprocess(x, v2=True):

    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


# In[29]:


# Creating of the metrics for model evaluation
def recall_m(actual, predicted):
    tp = K.sum(K.round(K.clip(actual * predicted, 0, 1)))
    positive_posible = K.sum(K.round(K.clip(actual, 0, 1)))
    recall = tp / (positive_posible + K.epsilon())
    return recall

def precision_m(actual, predicted):
    tp = K.sum(K.round(K.clip(actual * predicted, 0, 1)))
    positive_posible = K.sum(K.round(K.clip(predicted, 0, 1)))
    precision = tp / (positive_posible + K.epsilon())
    return precision

def f1_score(actual, predicted):
    precision = precision_m(actual, predicted)
    recall = recall_m(actual, predicted)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[30]:


def train_test_split():
    #Loading of dataset
    df = pd.read_csv("fer2013.csv")
    
    batch_size = 32
    num_epochs = 150
    input_shape = (48, 48, 1)
    verbose = 1
    num_classes = 7
    patience = 50
    path = 'models/'
    l2_regularization=0.01
    height = 48
    width = 48
    
    #dividing data into two format for training and test
    training_data = df[["emotion", "pixels"]][(df["Usage"] == "Training")]
    testing_data = df[["emotion", "pixels"]][(df["Usage"] == "PublicTest")]
    
    X_train,train_y,X_test,test_y=[],[],[],[]
    
    # Passing the respected data into X_train,train_y,X_test,test_y
    for x,y in training_data.iterrows():
        val=y['pixels'].split(" ")
        X_train.append(np.array(val,'float32'))
        train_y.append(y['emotion'])

    for index,row in testing_data.iterrows():
        val=row['pixels'].split(" ")
        X_test.append(np.array(val,'float32'))
        test_y.append(row['emotion'])
        
    #Changing the entire array of numpy array into float 32
    X_train = np.array(X_train,'float32')
    train_y = np.array(train_y,'float32')
    X_test = np.array(X_test,'float32')
    test_y = np.array(test_y,'float32')
    
    # Data normalization
    X_train = X_train - np.mean(X_train, axis=0)
    X_test = X_test - np.mean(X_test, axis=0)
    X_train = X_train / np.std(X_train, axis=0)
    X_test = X_test / np.std(X_test, axis=0)
    
    #Reshaping of data
    X_train = X_train.reshape(X_train.shape[0],48, 48,1)
    X_test = X_test.reshape(X_test.shape[0],48, 48 , 1)

    train_y = np_utils.to_categorical(train_y, num_classes=7)
    test_y = np_utils.to_categorical(test_y, num_classes=7)
    return X_train,train_y,X_test,test_y


# In[31]:


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:





# In[ ]:




