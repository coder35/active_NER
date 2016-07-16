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
indim = worddim

nerstr = "wordmodel"
if os.path.exists(nerstr):
    shutil.rmtree(nerstr)
os.mkdir(nerstr)
af = open(nerstr+"/data", 'wb')
pickle.dump(indexdata,af)
af.close()

df = open(nerstr+"/worddic", 'wb')
pickle.dump(wdic,df)
df.close()

tf = open(nerstr+"/tags",'wb')
pickle.dump(tags,tf)
tf.close()

drop_pro = 0.5
hidden_num  = 128
epoch_num = 14


#record location of every word in sentence
wordloc = []
for item in data:
    wordloc.append([])
    loc = 0
    for word in item[0]:
        wordloc[-1].append(loc)
        loc += len(word)
    wordloc[-1].append(loc)

testlocs = wordloc[len(indexdata)/20*18:len(indexdata)]

traindata = indexdata[0:len(indexdata)/20*16]
devdata = indexdata[len(indexdata)/20*16:len(indexdata)/20*18]
testdata = indexdata[len(indexdata)/20*18:len(indexdata)]

print "tain data number  "+str(len(traindata))+ "   dev   data  number  "+str(len(devdata))+"   test data  number "+str(len(testdata))

net_size = [indim, hidden_num*2, len(tags)]
print "model  initialization ............network size  "+str(net_size)
model = BLSTM.BLSTMmodel(len(tags),em_num, net_size,drop_pro, lrate=0.02, opt = "adaGad", embeddic = {"dic_1": embedding})
print "model training preparing.........................."
model.train_ready()
model.evaluate_ready()
print "model training ready ............................."
modelstr = nerstr+"/hidden_"+str(hidden_num)+"_drop_"+str(drop_pro)
if os.path.exists(modelstr):
    shutil.rmtree(modelstr)
os.mkdir(modelstr)

epoch = 1
trainscore = 0.

while True:
    print "epoch  .................................................................."+str(epoch)
    logging.debug("epoch  .................................................................."+str(epoch))
    traindata = preprocess.randomdata(traindata)
    #随机打乱数据
    print "generate batches ......."
    groups = preprocess.data2batch(traindata, batch_size,pids)

    print "start  update .............."
    for group in groups:
        #model.train_ready(group)
        groupscore = model.upAndEva(group)
        trainscore += groupscore

    print  "train score   :   "+str(trainscore)

    if epoch == 10:
        model.set_lrate(0.01)
    elif epoch == epoch_num:
        break
    epoch += 1

model.printmodel(modelstr+"/model")

wpredict = []
for item in testdata:
    wpredict.append(map(lambda x:x[0], model.decode(item[0], top_n)))


wresult1 = []
for item in wpredict:
    wresult1.append(pr.pres2en(tags, item))

#转化词语索引的(start, end, type) 为字符索引的(start, end, type)

wresult2 = []
for result,sentence in zip(wresult1,testlocs):
    wresult2.append([])
    for entity in result:
        wresult2[-1].append((sentence[entity[0]],sentence[entity[1]],entity[2]))



wresult = wresult2

# char model

data = preprocess.getcData(datafile)
cdic, cvectors = preprocess.chars2dic2("char_vector_50",chardim)
padding_id = cdic["<padding>"] = len(cvectors)
pids = [padding_id]
cvectors.append(np.random.randn(chardim))
embedding = cvectors
indexdata = preprocess.raw2num1(data,cdic,tags,0,padding_id)
indim = chardim
nerstr = "charmodel"
if os.path.exists(nerstr):
    shutil.rmtree(nerstr)
os.mkdir(nerstr)
af = open(nerstr+"/data", 'wb')
pickle.dump(indexdata,af)
af.close()

df = open(nerstr+"/chardic", 'wb')
pickle.dump(cdic,df)
df.close()


tf = open(nerstr+"/tags",'wb')
pickle.dump(tags,tf)
tf.close()


traindata = indexdata[0:len(indexdata)/20*16]
devdata = indexdata[len(indexdata)/20*16:len(indexdata)/20*18]
testdata = indexdata[len(indexdata)/20*18:len(indexdata)]

print "tain data number  "+str(len(traindata))+ "    dev   data  number  "+str(len(devdata))+"   test data  number "+str(len(testdata))

drop_pro = 0.4
hidden_num  = 160
epoch_num = 25

net_size = [indim, hidden_num*2, len(tags)]
print "model  initialization ............network size  "+str(net_size)
model = BLSTM.BLSTMmodel(len(tags),em_num, net_size,drop_pro, lrate=0.02, opt = "adaGad", embeddic = {"dic_1": embedding})
print "model training preparing.........................."
model.train_ready()
model.evaluate_ready()
print "model training ready ............................."

modelstr = nerstr+"/hidden_"+str(hidden_num)+"_drop_"+str(drop_pro)
if os.path.exists(modelstr):
    shutil.rmtree(modelstr)
os.mkdir(modelstr)

epoch  = 1
trainscore = 0
while True:
    print "epoch  .................................................................."+str(epoch)
    logging.debug("epoch  .................................................................."+str(epoch))
    traindata = preprocess.randomdata(traindata)
    #随机打乱数据
    print "generate batches ......."
    groups = preprocess.data2batch(traindata, batch_size,pids)

    print "start  update .............."
    for group in groups:
        #model.train_ready(group)
        groupscore = model.upAndEva(group)
        trainscore += groupscore

    if epoch == 10:
        model.set_lrate(0.01)
    elif epoch == epoch_num:
        break
    epoch += 1


model.printmodel(modelstr+"/model")

golden = []
cpredict = []
for item in testdata:
    cpredict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
    golden.append(item[1])

cresult = []
for item in cpredict:
    cresult.append(pr.pres2en(tags, item))

gold = []
for item in golden:
    gold.append(pr.pre2en(tags,item))


overall_p, overall_r, agreement, agree_p = pr.tri(gold,wresult,cresult)

print "overall precision  :   "+str(overall_p)+"  overall recall  :  "+ str(overall_r)+"   agreement rate  :  "+str(agreement)+"  agreement precision  :  "+str(agree_p)




