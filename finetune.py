# encoding=utf-8
import json
import random
import numpy as np
import preprocess
from semiBLSTM import semimodel
import os
import pickle
import shutil
import pr
import util
import sys
import gc
from time import gmtime, strftime
import io


def testmodel(testdata,sfilename,epoch,model,top_n,tags):
    predict = []
    gold = []
    print  "start  decoding   .............."
    for item in testdata:
        predict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
        gold.append(item[1])
    print  "decoding   finished  ..........."
    pr_dic = pr.computePR(tags, gold, predict)
    print  "test  result   :   "

    rev  = dict(zip(tags.values(), tags.keys()))
    
    score_file = open(sfilename,'ab')
    for key in pr_dic:
        precision = pr_dic[key][0]
        recall = pr_dic[key][1]
        f = 2./(1./(precision+1e-7)+1./(recall+1e-7))
        if key == "overall":
            overallf = f
            entype = "overall"
        else:
            entype = rev[key]
    
        s = "epoch  :  "+str(epoch)+"   entity  type  :  "+entype+"    p :  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
        print s
        score_file.write(s)
        
    score_file.close()
    return overallf





dim = 128
#cdic, cvectors = preprocess.chars2dic2("char_vector_50",dim)

storedir = sys.argv[1]

if os.path.exists(storedir):
    shutil.rmtree(storedir)
os.mkdir(storedir)


dropout_pro = 0.3
em_num = 1
top_n = 1

#get boson labeled data

batch_size = 10
data = preprocess.getwData("data.txt")
tags = preprocess.tags2dic(map(lambda x:x[1], data))

print "loading pretrained  word vectors ......................."
embeddic = pickle.load(open("pretrain_neg_allmask/premodel_epoch0_half_dic"))
wdic = pickle.load(open("pretrain_neg_allmask/tokendic"))
print "word vectors loading is done .............vector number "+str(len(embeddic["dic_1"]))
padding_id = wdic["<padding>"]
pids = [padding_id]


print "loading pretained model ..............................."
loadedmodel = pickle.load(open("pretrain_neg_allmask/premodel_epoch0_half_w"))


indexdata = preprocess.raw2num1(data,wdic,tags,0,padding_id)
traindata = indexdata[0:len(indexdata)/20*16]
devdata = indexdata[len(indexdata)/20*16:len(indexdata)/20*18]
testdata = indexdata[len(indexdata)/20*18:len(indexdata)]
'''

#get sighan labeled data                                                         
batch_size = 20                                                                                
traindata = preprocess.get_sighan_data("sighanNER/traindata")                    
devdata = preprocess.get_sighan_data("sighanNER/devdata")                        
testdata = preprocess.get_sighan_data("sighanNER/testdata")                      
tags = preprocess.tags2dic(map(lambda x:x[1], traindata))                        
 

print "loading pretrained  word vectors ......................."                 
embeddic = pickle.load(open("pretrain_mask2/premodel_epoch2_dic"))              
wdic = pickle.load(open("pretrain_mask2/tokendic"))                             
print "word vectors loading is done .............vector number "+str(len(embeddic["dic_1"]))
padding_id = wdic["<padding>"]                                                   
pids = [padding_id]                                                              

                                                                                  
print "loading pretained model ..............................."                  
loadedmodel = pickle.load(open("pretrain_mask2/premodel_epoch2_w"))             
print loadedmodel.keys()                                                         


traindata = preprocess.raw2num1(traindata,wdic,tags,0,padding_id)                
devdata = preprocess.raw2num1(devdata, wdic, tags,0,padding_id)                  
testdata = preprocess.raw2num1(testdata, wdic, tags,0,padding_id)       

'''

newloaded = {}
for key in loadedmodel:
    if key.startswith("w") or key.startswith("b"):
        newloaded[key] = loadedmodel[key]
loadedmodel = newloaded  


print loadedmodel.keys()

#list1 = range(100,110,10)#tune
#list2 = [0]*len(list1)#tune
#list1 = [100]*10
#list2 = range(5,55,5)
#for hidden_num, semiweight in zip(list1,list2):
list1 = [120]
list2 = [90]
for hidden_num1, hidden_num2 in zip(list1,list2):
        semiweight = 0
        #semiweight = semiweight/10.
        
	
	#net_size = [dim, hidden_num1*2, len(tags)]
	net_size = [dim, hidden_num1*2,hidden_num2*2, len(tags)]
        print "model  initialization ..............network size  "+str(net_size)
        model = semimodel(len(tags),em_num, net_size,dropout = dropout_pro, lrate=0.1, wdecay = 0.0, opt = "adaGrad",fix_emb = True, embeddic = embeddic, premodel = loadedmodel)


        print "model training preparing.........................."
        model.train_ready()
        model.evaluate_ready()
        print "model training ready ............................."
        modelstr = storedir+"/hid1_"+str(hidden_num1)+"_hid2_"+str(hidden_num2)
        if os.path.exists(modelstr):
            shutil.rmtree(modelstr)
        os.mkdir(modelstr)
        os.mknod(modelstr+"/trainscore")
        os.mknod(modelstr+"/devscore")


        print "start fine-tuning   .........."
        epoch  = 1
        bestf = 0.
        while True:
            print "epoch  .................................................................."+str(epoch)
            traindata = preprocess.randomdata(traindata)
            #随机打乱数据
            print "generate batches ......."
            supergroups = preprocess.data2batch(traindata,batch_size,pids)
            print "start  joint update .............."
            semiscore = 0
            #wikibatches = [padgroup(group,padding_id) for group in genAnEpoch(endata,wgroup_size,len(supergroups))]
            for sgroup in supergroups:
                #groupscore = model.semiUpdate(wikigroup, sgroup)
                groupscore = model.superUpdate(sgroup)

                print "group loss : "+str(groupscore)
                semiscore+=groupscore

            print " epoch   all   loss :  "+str(semiscore)
            print "update over  ....."

            #test on dev data
            print "...............test on development data "
            sfile = modelstr+"/devscore"
            model.set_dropout(0.)
            f = testmodel(devdata, sfile, epoch,model,top_n,tags)
            model.set_dropout(dropout_pro)
            if f > bestf and epoch > 3:
                model.printmodel(modelstr+"/model_epoch"+str(epoch),False)
                bestf = f
                sfile = modelstr+"/testscore"
                model.set_dropout(0.)
                testmodel(testdata, sfile, epoch, model,top_n,tags)
                model.set_dropout(dropout_pro)


            if epoch == 5 or epoch == 10:
                sfile = modelstr+"/trainscore"
                model.set_dropout(0.)
                testmodel(traindata, sfile, epoch,model,top_n,tags)
                model.set_dropout(dropout_pro)

            if epoch == 6:
                model.set_lrate(0.03)
            elif epoch == 11:
		model.set_lrate(0.015)
            elif epoch == 13:
                break
            epoch += 1


