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
mtype = "word"
bsize = 10

if mtype == "word":
    data = preprocess.getwData(datafile)
    tags = preprocess.tags2dic(map(lambda x:x[1], data))
    wdic, wvectors = preprocess.words2dic2("skip_neg10_50", worddim)
    padding_id = wdic["<padding>"] = len(wvectors)
    pids = [padding_id]
    wvectors.append(np.random.randn(worddim))
    embedding = wvectors
    indexdata = preprocess.raw2num1(data,wdic,tags,0,padding_id)
    indim = worddim

    nerstr = sys.argv[1]
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

elif mtype == "char":
    data = preprocess.getcData(datafile)
    tags = preprocess.tags2dic(map(lambda x:x[1], data))
    cdic, cvectors = preprocess.chars2dic2("char_vector_50",chardim)
    padding_id = cdic["<padding>"] = len(cvectors)
    pids = [padding_id]
    cvectors.append(np.random.randn(chardim))
    embedding = cvectors
    indexdata = preprocess.raw2num1(data,cdic,tags,0,padding_id)
    indim = chardim
    nerstr = sys.argv[1]
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


traindata = indexdata[0:len(data)/20*16]
devdata = indexdata[len(data)/20*16:len(data)/20*18]
testdata = indexdata[len(data)/20*18:len(data)]

print "tain data number  "+str(len(traindata))+ "    dev   data  number  "+str(len(devdata))+"   test data  number "+str(len(testdata))
for hidden_num in range(60,100,10):
        drop_pro = 0.
        # drop_pro = float(drop)/10.
        net_size = [indim, hidden_num*2, len(tags)]
        print "model  initialization ............network size  "+str(net_size)
        model = BLSTM.BLSTMmodel(len(tags),em_num, net_size,drop_pro, wdecay = 0, lrate=0.02, opt = "adaGrad", embeddic = {"dic_1": embedding})
        print "model training preparing.........................."
        model.train_ready()
        model.evaluate_ready()
        print "model training ready ............................."
        epoch  = 1
        testscores = []
        testscores.append(0)
        reduce_count = 0

        modelstr = nerstr+"/hid_"+str(hidden_num)
        if os.path.exists(modelstr):
            shutil.rmtree(modelstr)
        os.mkdir(modelstr)
        os.mknod(modelstr+"/trainscore")
        os.mknod(modelstr+"/devscore")
        while True:
            print "epoch  .................................................................."+str(epoch)
            logging.debug("epoch  .................................................................."+str(epoch))
            traindata = preprocess.randomdata(traindata)
            #随机打乱数据
            print "generate batches ......."
            groups = preprocess.data2batch(traindata, batch_size,pids)

            print "start  update .............."
            tscore = 0#training score
            count = 0
            for group in groups:
                #model.train_ready(group)
                groupscore = model.upAndEva(group)
                print "group  score  :  "+str(groupscore)
                tscore += groupscore

            print " ...........train data  score  :   "+str(tscore)
            #test for each epoch
            #model.set_dropout(0)

            #test on dev data
            predict = []
            gold = []
            for item in devdata:
                predict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
                gold.append(item[1])

            precision, recall = pr.computePR(tags, gold, predict)
            f = 2./(1./(precision+1e-8)+1./(recall+1e-8))
            score_file = open(modelstr+"/devscore",'ab')
            print  "develop  result   :   "
            s = "epoch  :  "+str(epoch) +" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
            print s
            score_file.write(s)
            score_file.close()

            #test on train data
            predict = []
            gold = []
            for item in traindata:
                predict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
                gold.append(item[1])


            precision, recall = pr.computePR(tags, gold, predict)
            f = 2./(1./(precision+1e-8)+1./(recall+1e-8))
            score_file = open(modelstr+"/trainscore",'ab')
            print  "test result   :   "
            s = "epoch  :  "+str(epoch) +" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
            print s
            score_file.write(s)
            score_file.close()

            #model.set_dropout(drop_pro)

            if epoch == 10:
                model.set_lrate(0.01)
            elif epoch == 20:
                break
            epoch += 1




