# coding=utf-8
import preprocess
import pickle
import BLSTM
import os
import random
import shutil
import logging
import pr
import numpy as np
import sys

logging.basicConfig(level = logging.DEBUG,
                    filename = 'modelrun.log',
                    filemode = 'w')
batch_size = 20
cdic_size = 5000
worddim = 50
chardim = 50
datafile= "data.txt"
window_size = 0
top_n = 1
em_num = 1
bsize = 10




# word model train and test
data = preprocess.getwData(datafile)
tags = preprocess.tags2dic(map(lambda x:x[1], data))
wdic, wvectors = preprocess.words2dic2("skip_neg10_50", worddim)
padding_id = wdic["<padding>"] = len(wvectors)
pids = [padding_id]
wvectors.append(np.random.randn(worddim))
embedding = wvectors
indexdata = preprocess.raw2num1(data,wdic,tags,0,padding_id)


#record location of every word in sentence
wordloc = []
for item in data:
    wordloc.append([])
    loc = 0
    for word in item[0]:
        wordloc[-1].append(loc)
        loc += len(word)
    wordloc[-1].append(loc)

for word,loc in zip(data[0][0],wordloc[0]):
    print word.encode('utf-8')+str(loc)

testlocs = wordloc[len(indexdata)/20*18:len(indexdata)]

traindata = indexdata[0:len(indexdata)/20*16]
devdata = indexdata[len(indexdata)/20*16:len(indexdata)/20*18]
testdata = indexdata[len(indexdata)/20*18:len(indexdata)]



wresult1 = []
for item in testdata:
    wresult1.append(pr.pre2en(tags, item[1]))

#转化词语索引的(start, end, type) 为字符索引的(start, end, type)

wresult2 = []
for result,sentence in zip(wresult1,testlocs):
    wresult2.append([])
    print sentence
    for entity in result:
        print entity
        wresult2[-1].append((sentence[entity[0]], sentence[entity[1]], entity[2]))



wresult = wresult2

# char model

data = preprocess.getcData(datafile)
cdic, cvectors = preprocess.chars2dic2("char_vector_50",chardim)
padding_id = cdic["<padding>"] = len(cvectors)
pids = [padding_id]
cvectors.append(np.random.randn(chardim))
embedding = cvectors
indexdata = preprocess.raw2num1(data,cdic,tags,0,padding_id)

traindata = indexdata[0:len(indexdata)/20*16]
devdata = indexdata[len(indexdata)/20*16:len(indexdata)/20*18]
testdata = indexdata[len(indexdata)/20*18:len(indexdata)]


testwdata = data[len(indexdata)/20*18:len(indexdata)]

cgold = []
for item in testdata:
    cgold.append(pr.pre2en(tags,item[1]))


error = 0
isum = 0
lines = []
iid = 0
for item1, item2 in zip(wresult, cgold):
    for en1, en2 in zip(item1,item2):
        isum += 1
        if en1[0] != en2[0] or en1[1] != en2[1] or en1[2] != en2[2]:
            error+=1
            lines.append(iid)
            break
    iid += 1

for iid in lines:
    print " ".join(testwdata[iid][0])
    for en1,en2 in zip(wresult[iid],cgold[iid]):
        print "".join(testwdata[iid][0][en1[0]:en1[1]])+"    "+ "".join(testwdata[iid][0][en2[0]:en2[1]])


print "sum  :  "+str(isum)+"  error  :  "+str(error)



