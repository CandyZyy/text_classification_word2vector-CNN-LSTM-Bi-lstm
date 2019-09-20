#coding=utf-8
import numpy as np
import os
import pandas as pd
import nltk

#生成、统计讨论预料库中的单个词语
data=pd.read_csv("C:\\Users\\ZYY\\Desktop\\data_process.csv",usecols=['text','class','clean_text'],encoding='utf-8')
def split_sentences(descri):
    sentence = descri.split(',')
    sentence = sentence[0:50]
    return sentence
sentences=sum((data['clean_text'].apply(split_sentences)),[])
corpus=list(set(sentences))
np.savetxt("C:\\Users\\ZYY\\Desktop\\allvocabs.txt",corpus,fmt='%s',delimiter='\n')
print("整理成句子处理文本的词汇数：",len(corpus))
fq=nltk.FreqDist(sentences)
print(fq)
print(fq.most_common(30))