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



dim = 128
#cdic, cvectors = preprocess.chars2dic2("char_vector_50",dim)

storedir = sys.argv[1]

if os.path.exists(storedir):
    shutil.rmtree(storedir)
os.mkdir(storedir)


em_num = 1
top_n = 120 
beam_size = 250

#get boson labeled data

batch_size = 10
data = preprocess.getwData("data.txt")
tags = preprocess.tags2dic(map(lambda x:x[1], data))

print "loading pretrained  word vectors ......................."
embeddic = pickle.load(open("finetune_epoch2_large_mask0.3_fixem/model_dic"))
wdic = pickle.load(open("finetune_epoch2_large_mask0.3_fixem/tokendic"))


print "word vectors loading is done .............vector number "+str(len(embeddic["dic_1"]))
padding_id = wdic["<padding>"]
pids = [padding_id]


print "loading pretained model ..............................."
loadedmodel = pickle.load(open("finetune_epoch2_large_mask0.3_fixem/model_epoch11_w"))


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



print loadedmodel.keys()

#list1 = range(100,110,10)#tune
#list2 = [0]*len(list1)#tune
#list1 = [100]*10
#list2 = range(5,55,5)
#for hidden_num, semiweight in zip(list1,list2):
list1 = [100]
list2 = [70]
for hidden_num1, hidden_num2 in zip(list1,list2):
       
        
	
	#net_size = [dim, hidden_num1*2, len(tags)]
	net_size = [dim, hidden_num1*2,hidden_num2*2, len(tags)]
        print "model  initialization ..............network size  "+str(net_size)
        model = semimodel(len(tags),em_num, net_size,dropout = 0, lrate=0.1, wdecay = 0.0, opt = "adaGrad",fix_emb = True, embeddic = embeddic, premodel = loadedmodel)

        print "model evaluating preparing.........................."
        
        model.evaluate_ready()
        print "model training ready ............................."
        modelstr = storedir+"/hid1_"+str(hidden_num1)+"_hid2_"+str(hidden_num2)
        if os.path.exists(modelstr):
            shutil.rmtree(modelstr)
        os.mkdir(modelstr)
        os.mknod(modelstr+"/testscore")
        
        

        predict = []
        gold = []
        print  "start  decoding   .............."
        for item in testdata:
            #predict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
            predict.append(map(lambda x:x[0], model.beamsearch(item[0], beam_size, top_n)))
            
            gold.append(item[1])
        print  "decoding   finished  ..........."
        pr_dic = pr.computePR(tags, gold, predict)
        print  "test  result   :   "

        rev  = dict(zip(tags.values(), tags.keys()))
        
        sfilename = modelstr+"/testscore"
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
    
            s = "entity  type  :  "+entype+"    p :  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
            print s
            score_file.write(s)
        
        score_file.close()
               
