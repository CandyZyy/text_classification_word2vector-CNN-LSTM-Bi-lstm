import numpy as np
import pandas as pd
from keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler
from collections import Counter

def resample(x,y):
    #法一：过采样
    # ros=RandomOverSampler(random_state=0)
    # x_resampled,y_resampled=ros.fit_resample(x,y)
    #法二：下采样
    # cc=ClusterCentroids(random_state=0)
    # x_resampled, y_resampled = cc.fit_resample(x, y)

    # rus=RandomUnderSampler(random_state=0)
    # x_resampled, y_resampled = rus.fit_resample(x, y)
    #法三：过采样和下采样结合
    # smote_enn=SMOTEENN(random_state=0)
    # x_resampled, y_resampled = smote_enn.fit_resample(x, y)

    smote_tomek=SMOTETomek(random_state=0)
    x_resampled, y_resampled = smote_tomek.fit_resample(x, y)
    return x_resampled,y_resampled

#定义数据集
# data=pd.read_csv("/home/zyy/桌面/data_process.csv",usecols=['text','class','clean_text'])
def data_split():
    data_text=np.load('/home/zyy/桌面/idMatrix23.npy')
    data_lable=pd.read_csv('/home/zyy/桌面/class_id.csv')
    np.save('/home/zyy/桌面/class_id.npy',data_lable)
    data_lable=np.load('/home/zyy/桌面/class_id.npy',allow_pickle=True)
    x_train=[data_text[0:12836].copy()]
    x_train_list=data_text[0:12836].copy()
    y_train=data_lable[0:12836]
    y_train=to_categorical(y_train)
    x_train_list,y_train=resample(x_train_list,y_train)
    x_test=[data_text[12836:].copy()]
    x_test_list=data_text[12836:].copy()
    y_test=data_lable[12836:]
    y_test=to_categorical(y_test)
    #避免验证集准确率很低，提前将训练集的数据对应打乱
    np.random.seed(200)
    np.random.shuffle(x_train_list)
    np.random.seed(200)
    np.random.shuffle(y_train)
    np.random.seed(200)
    np.random.shuffle(x_test_list)
    np.random.seed(200)
    np.random.shuffle(y_test)
    # y_train_max=np.argmax(y_train,axis=1)
    # np.save('/home/zyy/桌面/oversample.npy', y_train_max)
    return x_train,x_train_list,y_train,x_test,x_test_list,y_test

#定义词向量
def voca_vec():
    vec=np.load('/home/zyy/桌面/embedrealvec.npy',allow_pickle=True)
    return vec