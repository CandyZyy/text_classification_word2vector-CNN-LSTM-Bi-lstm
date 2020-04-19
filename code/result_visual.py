import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np

def plot_history(history,name):
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    # # 绘制训练 & 验证的损失值
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    accurac=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    x=range(1,len(accurac)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(x,accurac,'b',label='Training acc')
    plt.plot(x,val_acc,'r',label='Validation acc')
    plt.title('Training and validation accracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(x,loss,'b',label='Training loss')
    plt.plot(x,val_loss,'r',label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("../result/str(name)+.png")
    plt.show()


def classify_standard(y_test, y_predict):

    t=classification_report(y_test, y_predict,
                         target_names=['class1', 'class2', 'class3', 'class4', 'class5'],
                         digits=4)
    print(t)


def plot_confusion_matrix(y_test, y_predict, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_test, y_predict)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax