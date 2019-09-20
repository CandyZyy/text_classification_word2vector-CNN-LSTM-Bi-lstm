import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding,Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from result_visual import plot_history,plot_confusion_matrix,classify_standard
from data_set_split import voca_vec,data_split
from sklearn.metrics import classification_report
from train_model import train_model_process


#定义数据集
x_train,x_train_list,y_train,x_test,x_test_list,y_test=data_split()
x_train=x_train_list
x_test=x_test_list

#定义词向量
maxlen=50
embed_dim=300
vocab_size=7501
vec=voca_vec()

#定义模型
print("Building model......")
num_class=5
batch_size=128
model=Sequential()
model.add(Embedding(vocab_size,embed_dim,weights=[vec],input_length=maxlen,trainable=True))
model.add(LSTM(256,dropout=0.2,return_sequences=False))
model.add(Dense(10,activation="relu"))
model.add(Dense(5,activation="softmax"))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#模型训练
epochs=200
history,y_true_score,y_pred_score=train_model_process(model,x_train,y_train,x_test,y_test,batch_size,epochs)
#结果展示
classify_standard(y_true_score,y_pred_score)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
classes = np.array([0, 1, 2, 3, 4])
plot_confusion_matrix(y_true_score,y_pred_score,classes=classes,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_true_score,y_pred_score, classes=classes,normalize=True,
                      title='Normalized confusion matrix')
plt.show()
plot_history(history,name='lstm')