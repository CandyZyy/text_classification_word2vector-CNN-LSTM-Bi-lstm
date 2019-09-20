import numpy as np
import os
import pandas as pd
import nltk

#取出大规模语料库中包含的讨论语料的词及对应的词向量
corpus=np.loadtxt("C:\\Users\\ZYY\\Desktop\\allvocabs.txt",dtype=str)
id=np.load("C:\\Users\\ZYY\\Desktop\\idvec.npy",allow_pickle=True).item()
embed=np.load("C:\\Users\\ZYY\\Desktop\\embedvec.npy",allow_pickle=True)
print(len(corpus))
print(embed.shape)

idvoc={}
embedvoc=[]
nonvec=[]
start=0
for i, voc in enumerate(corpus):
        if (voc in id) ==True :
            idvoc[voc]=start
            em=embed[id[voc]].copy()
            em_new=[float(em[t]) for t in range(0, 300)]
            embedvoc.append(em_new)
            start=start+1
        else:
            nonvec.append(voc)

idvoc["not in bigvoca"]=7500
lastlist =[]
for i in range(300):
    lastlist.append(0.00)
embedvoc.append(lastlist)
embedvoc=np.array(embedvoc,dtype=np.float32)
print(embedvoc.dtype,embedvoc.shape)
np.save("C:\\Users\\ZYY\\Desktop\\embedrealvec.npy",embedvoc,)
print("ok")
# np.save("C:\\Users\\ZYY\\Desktop\\nonvec.txt",nonvec)
np.savetxt("C:\\Users\\ZYY\\Desktop\\nonvec.txt",nonvec,fmt='%s',delimiter='\n')
print("ok")
np.save("C:\\Users\\ZYY\\Desktop\\idrealvec.npy",idvoc)
print("ok")

embedvoc=np.load("C:\\Users\\ZYY\\Desktop\\embedrealvec.npy",allow_pickle=True)
print("词向量维度",embedvoc.shape,type(embedvoc))
idvoc=np.load("C:\\Users\\ZYY\\Desktop\\idrealvec.npy",allow_pickle=True).item()
print("包含处理文本词汇数：",len(idvoc),type(idvoc))
t=idvoc.values()
print(max(t))
# 词索引组成句子向量
maxSeqLength=50
numDimentions=60
senCounter=0
length=16047
ids=np.zeros((length,maxSeqLength),dtype='int32')
data=pd.read_csv("C:\\Users\\ZYY\\Desktop\\data_process.csv",usecols=['text','class','clean_text'],encoding='utf-8')
def split_sentences(descri):
    sentence = descri.split(',')
    sentence = sentence[0:50]
    sentence = [sentence]
    return sentence
sentences=sum((data['clean_text'].apply(split_sentences)),[])
for sentence in sentences:
    indexCounter = 0
    for word in sentence:
        try:
            ids[senCounter][indexCounter]=idvoc[word]
        except KeyError:
            ids[senCounter][indexCounter]=7500
        indexCounter = indexCounter + 1
    senCounter=senCounter+1
np.save('C:\\Users\\ZYY\\Desktop\\idMatrix23.npy',ids)
print("ok")
