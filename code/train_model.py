import numpy as np
import keras.callbacks as kcallbacks

def train_model_process(model,x_train,y_train,x_test,y_test,batch_size,epochs):
    #模型训练
    print("Training......")
    earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    history=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
                      validation_split=0.2,shuffle=True,callbacks = [earlyStopping],
                      verbose=2)

    loss_train, acc_train = model.evaluate(x_train,y_train, batch_size=batch_size,verbose=0)
    print('Train loss:', loss_train)
    print('Train accuracy:', acc_train)
    loss, acc = model.evaluate(x_test,y_test, batch_size=batch_size,verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    print(model.summary())
    y_predict=model.predict(x_test)
    y_pred_score = np.argmax(y_predict, axis=1)  # 反one-hot编码
    y_true_score = np.argmax(y_test, axis=1)  # 反one-hot编码
    # np.save('/home/zyy/桌面/y_true',y_true_score)
    # np.save('/home/zyy/桌面/y_pred',y_pred_score)
    return history,y_true_score,y_pred_score

