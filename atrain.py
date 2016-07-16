# encoding=utf-8
import numpy as np
import myio
from token_model import *
import sys
import pickle
import json
import random
from util import *
from asample import *



if __name__ == "__main__":

  pool_file = sys.argv[1]
  prefile = sys.argv[3]
  model_id = sys.argv[4]


  batch_size = 128
  context_len = 20
  model_type = "softmax"
  #net_size = [150,360,250,6]
  net_size = [150, 440,300,6]
  


  drop_pro = 0.45


  wdecay = 1e-5
  conreg_weight1 = 0.1
  
  conreg_weight2 = 0.01
  variance = 0.05

  #conreg_weight2 = 0.1
  #variance = 0.05
  
  #conreg_weight2 = 1
  #variance = 0.05




  print "loading data......................"
  embeddings= pickle.load(open("charembeddings")) # embedding matrix 
  tokendic = pickle.load(open("chardic"))# (word : index)
  padding_id = tokendic["<padding>"]
  padding_y = -1


  #tagdic = pickle.load(open("weibo_data/labeldic"))
  #neg_label = tagdic["O"]
  #print len(tagdic)
  #assert neg_label == len(tagdic) - 1
  
  #traindata = pickle.load(open("weibo_data/traindata"))
  #devdata = pickle.load(open("weibo_data/devdata"))
  #testdata = pickle.load(open("weibo_data/testdata"))




  tagdic = pickle.load(open("tagdic"))
  neg_label = tagdic["NONE"]
  assert neg_label == len(tagdic) - 1
  traindata = pickle.load(open("cleandata/traindata"))
  devdata = pickle.load(open("cleandata/devdata"))
  testdata = pickle.load(open("cleandata/testdata"))
  pooldata = pickle.load(open(pool_file))

  #double traindata
  #traindata = traindata + pooldata[:len(traindata)]

  print "traindata ins num :  "+str(len(traindata))+"  devdata ins  num  : "+str(len(devdata))+"  testdata  ins  num  :  "+str(len(testdata))+ "   pooldata  ins num :  "+str(len(pooldata))
  
  #bound 
  traindata = traindata+pooldata


  tdata_pos, tdata_neg = myio.create_ins(traindata, context_len, neg_label, padding_id, padding_y)

  print "from train........pos data number  :  "+str(len(tdata_pos))+"  neg   data  number  :  "+str(len(tdata_neg))
  tdata1 = tdata_pos + tdata_neg
  tdata2 = pool2ins(pooldata, padding_id, context_len)
  print "from pool ............ data numer  :  "+str(len(tdata2))

  

  tdata = tdata1 + tdata2

  
  print "all  training data num................ "+str(len(tdata))
  
  premodel = {}
  if prefile != "none":
    imodel = pickle.load(open(prefile))
    premodel = imodel["model_weight"]
    net_size =  imodel["net_size"]
    drop_pro = imodel["dropout"]
    net_size[-1] = 6
    print "load  premodel  ...................."+prefile
  if model_id == "aut":
    midstr = "LSTM_forward"
  elif model_id == "axu":
    midstr = "LSTM_LSTM"
    myio.masktarget(tdata, padding_id)
    print "padding  id  :  "+str(padding_id)
    print tdata[:2]
	  
  print "model  size.............."+str(net_size)
  print "model drop out .............. "+str(drop_pro)
  model = tokenModel(neg_label = neg_label,net_size = net_size,midstr = midstr, embeddic = embeddings, mid_id = context_len, conreg_weight1 = conreg_weight1, conreg_weight2 = conreg_weight2, variance = variance,
    drop_pro = drop_pro , model_type = model_type,premodel = premodel, pos_ratio = 0.5, wdecay = wdecay, mo_wdecay = 0.02, opt = "adaGrad", wbound = -0.4, fix_emb = True, ismention = False, istransition = False)
  model.train_ready()
  model.evaluate_ready()
   
  para = {}
  para["batch_size"] = batch_size 
  if model_type == "softmax_reg":
    para["score_dir"] = sys.argv[2]+"_model_"+model_type+"_"+str(net_size)+"_drop"+str(drop_pro)+"_wdecay"+str(wdecay)+"_weight1"+str(conreg_weight1)+"_weight2"+str(conreg_weight2)+"_variance"+str(variance)
  else:
    para["score_dir"] = sys.argv[2]+"_model_"+model_type+"_"+str(net_size)+"_drop"+str(drop_pro)+"_wdecay"+str(wdecay)
  #else:
    #para["score_dir"] = testdir+"/"+testfile+"_test"
  #para["posnum"] = len(tdata_pos)
  #para["negnum"] = len(tdata_neg)
  para["posnum"] = 1
  para["negnum"] = 1

  #para["lrates"] = [0.001, 0.001, 0.001,0.001,0.001, 0.001,0.001,0.001] #language model training [150,500,360,5000]
  #para["lrates"] = [0.01, 0.01, 0.01,0.01,0.01, 0.01,0.002,0.002,0.002,0.002,0.002,0.002] #maxout training [150,480,350,(5,)]
  #para["lrates"] = [0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
  #para["lrates"] = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
  
  #para["lrates"] = [0.01, 0.01, 0.003,0.003,0.001,0.001,0.001,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003] #maxout training [150,600,430,(5,)]
  #para["lrates"] = [0.04, 0.008, 0.008,0.008, 0.008,0.002,0.002,0.002,0.002,0.002,0.002] #max neg training [150,480,350,(5,)] 

  if model_id == "aut":
  	#para["lrates"] = [0.04,0.008,0.002,0.002,0.002,0.002,0.002,0.002] # softmax training [150, 360, 250, 6]round0
    #para["lrates"] = [0.02]*6 # softmax training [150, 360, 250, 6]round0
    #para["lrates"] = [0.02]*3+[0.01]*3 # softmax training [150, 360, 250, 6]round0
    para["lrates"] = [0.04]*10 # softmax training [150, 360, 250, 6]round0
  	#para["lrates"] = [0.02,0.008,0.002,0.002,0.002,0.002,0.002,0.002] # softmax training [150, 360, 250, 6]round0
  
  elif model_id == "axu":
  	para["lrates"] = [0.02,0.002,0.002,0.002,0.002,0.002,0.002,0.002] # softmax training [150, 360, 250, 6]
  	#para["lrates"] = [0.01,0.004,0.004,0.002,0.002,0.002,0.002,0.002] # softmax training [150, 360, 250, 6]
  #para["lrates"] = [0.01,0.002,0.001,0.001,0.001,0.001,0.001,0.001,0.001] # softmax training [150, 500,360 , 6]
  #para["lrates"] = [0.01,0.002,0.001,0.001,0.001,0.001,0.001,0.001,0.001] # softmax training [150, 500,360 , 6]
  #para["lrates"] = [0.02,0.006,0.006,0.006,0.002,0.002,0.002,0.002] # softmax training [150, 360, 250, 6]round1
  #para["lrates"] = [0.02,0.004,0.002,0.002,0.002,0.001,0.001,0.001,0.001] # softmax training [150, 500,360 , 6]
  #para["lrates"] = [0.02,0.004,0.004,0.002,0.002,0.002,0.002,0.002] # softmax training [150, 360, 250, 6] 2round
  
  #test
  #para["lrates"] = [0.04]





 
  model.train(tdata, traindata, devdata, pooldata, para, testdata)




 
