import numpy as np
from keras.models import Sequential
from keras.layers import Embedding,Dense,LSTM,Bidirectional,Dropout
from data_set_split import voca_vec,data_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from result_visual import plot_history,plot_confusion_matrix,classify_standard
from train_model import train_model_process
import matplotlib.pyplot as plt


x_train,x_train_list,y_train,x_test,x_test_list,y_test=data_split()
x_train=x_train_list
x_test=x_test_list
maxlen = 50
embed_dim = 300
vocab_size = 7501
vec=voca_vec()

def creat_model(trainable):
    model=Sequential()
    model.add(Embedding(vocab_size,embed_dim,weights=[vec],input_length=maxlen,trainable=trainable))
    model.add(Dropout(0.7))
    model.add(Bidirectional(LSTM(100, dropout=0.7,return_sequences=True)))

    model.add(Bidirectional(LSTM(100,dropout=0.7)))
    model.add(Dense(5,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#超参数
batch_size=64
'''
output_file='/home/zyy/桌面/CNN/cnn_parameters_output.txt'

# 定义参数空间，进行参数选择
param_grid=dict(num_filters=[16,32,64,128,256],
                kernel_size=[3,5,7],
                trainable=[True,False])
model=KerasClassifier(build_fn=creat_model,epochs=epochs,batch_size=batch_size,verbose=2)
#cv为做三折的交叉验证
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=2,verbose=2)
grid_result=grid.fit(x_train,y_train)
test_accuracy=grid.score(x_test,y_test)
with open(output_file,'a')as f:
    s=('Best Accuracy:' '{:.4f}\n{}\nTest Accuracy:{:.4f}\n\n')
    output_string=s.format(grid_result.best_score_,
                           grid_result.best_params_,
                           test_accuracy)
    print(output_string)
    f.write(output_string)
print("Ending.....")


'''
#模型训练
epochs=200
model=creat_model(True)
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
plot_history(history,name='Bi-lstm')
