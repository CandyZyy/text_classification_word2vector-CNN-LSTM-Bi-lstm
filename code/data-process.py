import re
import jieba
import numpy as np
import pandas as pd


# data=pd.read_csv("/home/zyy/桌面/data.csv",usecols=['text','class'])
data=pd.read_csv("../data/data.csv",usecols=['text','class'])
def clean_text(texts):
    texts=str(texts)
    words=list(jieba.cut(texts,cut_all=False))
    words = ','.join(words)
    return words
data['clean_text']=data.text.apply(clean_text)
data.to_csv("../data/data_process.csv")
np.seterr(invalid='ignore')

