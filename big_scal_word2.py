#coding=utf-8

import numpy as np
import os
import pandas as pd
import nltk

# String embedFile
def readEmbedFile(embedFile):
    embedId = {}
    input = open(embedFile,'r',encoding="utf-8")
    lines = []
    a=0
    for line in input:
        lines.append(line)
        a=a+1
        print(a)
    nwords = len(lines) - 1
    splits = lines[1].strip().split(' ')  # 因为第一行是统计信息，所以用第二行
    dim = len(splits) - 1
    embeddings=[]
    # embeddings = [[0 for col in range(dim)] for row in range(nwords)]
    b=0
    for lineId in range(len(lines)):
        b=b+1
        print(b)
        splits = lines[lineId].split(' ')
        if len(splits) > 2:
            # embedId赋值
            embedId[splits[0]] = lineId
            # embeddings赋值
            emb = [float(splits[i]) for i in range(1, 301)]
            embeddings.append(emb)
    return embedId, embeddings

id,embed=readEmbedFile("C:\\Users\\ZYY\\Desktop\\vocabvec.txt")
np.save("C:\\Users\\ZYY\\Desktop\\embedvec.npy",embed)
print("ok")
np.save("C:\\Users\\ZYY\\Desktop\\idvec.npy",id)
print("ok")