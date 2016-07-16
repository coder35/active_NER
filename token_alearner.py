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
import re



if __name__ == "__main__":
  ite = 10
  batch_size = 128
  context_len = 20
  neglabel_num = 10
  
  #model_type = "softmax"
  #net_size = [150,500,360,5001]
  #print "training  language model........"
  
  #model_type = "softmax"
  #net_size = [150,440,300,4001]
  #print "training  language model........"
  
  #model_type = "softmax"
  #net_size = [150,360,250,6]

  model_type = "softmax"
  #net_size = [150,440,300,6]

  net_size = [150,380,250,6]


  #model_type = "softmax"
  #net_size = [150,500,360,6]


  wdecay = 1e-5

  conreg_weight1 = 0.1
  
  conreg_weight2 = 0.05
  variance = 0.05
  nc = 10


  midstr = "LSTM_forward"
  mid_id = context_len
  premodel = {}
  drop_pro = 0.6

  active_type = sys.argv[1]
  sample_num = int(sys.argv[2])
  poolfile = sys.argv[3]
  bestmodel = sys.argv[4]
  if active_type == "LC":
    active_learner = learner_LC(sample_num)
    print "adopt  least confidence  active  learner ........"
  elif active_type == "margin":
    active_learner = learner_margin(sample_num)
    print "adopt  margin  based  active  learner  ........"
  elif active_type == "entropy":
    active_learner = learner_entropy(sample_num)
    print  "adopt  entropy  based   active  learner ........"
  elif active_type == "random":
    active_learner = baseline_random(sample_num)
    print "random  sampling .............."


  print "loading data......................"
  embeddings= pickle.load(open("charembeddings")) # embedding matrix 
  tokendic = pickle.load(open("chardic"))# (word : index)
  padding_id = tokendic["<padding>"]
  padding_y = -1



  tagdic = pickle.load(open("tagdic"))
  neg_label = tagdic["NONE"]
  assert neg_label == len(tagdic) - 1


  traindata = pickle.load(open("cleandata/train70k/traindata"))
  devdata = pickle.load(open("cleandata/train70k/devdata"))
  testdata = pickle.load(open("cleandata/train70k/testdata"))


  #traindata = pickle.load(open("cleandata/train70k/traindata"))
  #devdata = pickle.load(open("cleandata/train70k/devdata"))
  #testdata = pickle.load(open("cleandata/train70k/testdata"))
 
 
  #traindata = pickle.load(open("cleandata/train35k/traindata"))
  #devdata = pickle.load(open("cleandata/train35k/devdata"))
  #testdata = pickle.load(open("cleandata/train35k/testdata"))
  pooldata = pickle.load(open(poolfile))
  #pooldata = pickle.load(open("0_model_softmax_LC_[150, 360, 250, 6]/pooldata_best"))

  

  tdata_pos, tdata_neg = myio.create_ins(traindata, context_len, neg_label, padding_id, padding_y)

  print "from train........pos data number  :  "+str(len(tdata_pos))+"  neg   data  number  :  "+str(len(tdata_neg))
  tdata1 = tdata_pos + tdata_neg



 

  for ite in xrange(13):
    
    
   
    '''
    pool_pos = []
    pool_neg = []
    for item in tdata2:
      y = item[1]
      assert y[len(y)/2] != -1
      if y[len(y)/2] == neg_label:
        pool_neg.append(item)
      else:
        pool_pos.append(item)

    print "neg num in select data  :  "+str(len(pool_neg))+"   pos num in select data  :  "+str(len(pool_pos))


    all_posdata, _ = myio.create_ins(pooldata, context_len, neg_label, padding_id, padding_y)
    randomdata(all_posdata)

    pool_pos = all_posdata[:len(pool_pos)]
    tdata2 = pool_neg + pool_pos
    print "random from pool ............ data numer  :  "+str(len(tdata2))
    '''
    
    
    
    #myio.masktarget(tdata, padding_id)
    #print "padding  id  :  "+str(padding_id)
    #print tdata[:2]

    if bestmodel != "none":
      imodel = pickle.load(open(bestmodel))
      premodel = imodel["model_weight"]
      net_size =  imodel["net_size"]
      drop_pro = imodel["dropout"] - 0.005


    '''
    bestepoch = get_bestepoch("quarterdata/LC/round"+str(ite)+"/devscore")
    best_file = "quarterdata/LC/round"+str(ite)+"/modelfile_"+str(bestepoch)
 
    imodel = pickle.load(open(best_file))
    premodel = imodel["model_weight"]
    net_size =  imodel["net_size"]
    drop_pro = imodel["dropout"] - 0.005
    '''
    print "drop  out pro  ............. "+str(drop_pro)
    print "model net size .......... .... "+str(net_size)
        
    model = tokenModel(neg_label = neg_label,net_size = net_size,midstr = midstr, embeddic = embeddings,mid_id = context_len, conreg_weight1 = conreg_weight1, conreg_weight2 = conreg_weight2, variance = variance,nc =nc, drop_pro = drop_pro,
     model_type = model_type, premodel = premodel, pos_ratio = 0.5, wdecay = wdecay, mo_wdecay = 0.02, opt = "adaGrad", wbound = -0.4, fix_emb = True)
    model.train_ready()
    model.evaluate_ready()

    if bestmodel != "none":
      select = baseline_active_sample(model, pooldata, active_learner, isvote = False )
      tdata2 = pool2ins(pooldata, padding_id, context_len)

    #  print "sample negative  number ................ "+str(sample_num)
    #  select = random_active_sample(sample_num *2,sample_num, pooldata, neg_label = 5, israndom_neg = False, model = model, active_learner = active_learner)





    #pooldata = pickle.load(open("quarterdata/LC/round"+str(ite)+"/pooldata_best"))
    #randompos_active_sample(model, pooldata, active_learner, neg_label = 5)
    #tdata2 = pool2ins(pooldata, padding_id, context_len)

    


    print "traindata ins num :  "+str(len(traindata))+"  devdata ins  num  : "+str(len(devdata))+"  testdata  ins  num  :  "+str(len(testdata))+ "   pooldata  ins num :  "+str(len(pooldata))
    
    '''
    handfile = "tradition_baseline/halfdata/round"+str(ite+1)+"/pooldata"
    print "adop hand-crafted  pool data  file ............... "+handfile
    handpool = pickle.load(open(handfile))
    tdata2 = handpool2ins(pooldata, padding_id, context_len,handpool["pool_mark"])
    '''

    


   
    print "from pool ............ data numer  :  "+str(len(tdata2))
    tdata = tdata1 + tdata2
    print "all  training data num :  "+str(len(tdata))+"  epoch   :  "+str(ite)


    para = {}
    para["batch_size"] = batch_size 
    if model_type == "softmax_reg":
      para["score_dir"] = sys.argv[5]+str(ite)+"_model_"+model_type+"_"+str(net_size)+"_drop"+str(drop_pro)+"_weigh1"+str(conreg_weight1)+"_weigh2"+str(conreg_weight2)+"_variance"+str(variance)+"_nc"+str(nc)
    else:
      para["score_dir"] = sys.argv[5]+str(ite)+"_model_"+model_type+"_"+str(net_size)+"_drop"+str(drop_pro)
   
     
   
    
    para["posnum"] = 1
    para["negnum"] = 1
    para["lrates"] = [0.02]*5


    bestmodel = model.train(tdata, traindata, devdata, pooldata, para, testdata)
    

    #added = pool2ins(pooldata, padding_id, context_len)
    

    


  #test
  '''  
  print "loading  model ............"
  model_para = pickle.load(open("softmax_baseline/modelfile_6"))
  model = tokenModel(model_para["neg_label"], model_para["net_size"],"LSTM_forward", embeddings, model_para["dropout"], model_para["model_type"], model_para["model_weight"])
  model.evaluate_ready()
  print "evaluate ready   is  done ............"
  #select1 = active_sample(model, pooldata, active_learner)
  model_para = pickle.load(open( "mask_2BLSTM_model_softmax_LC_[150, 500, 360, 6]/modelfile_1"))

  model_axu = tokenModel(model_para["neg_label"], model_para["net_size"], "LSTM_LSTM", embeddings, model_para["dropout"], model_para["model_type"], model_para["model_weight"])
  model_axu.evaluate_ready()
  ''' 


  '''
  resetpool(pooldata)


  print "loading  model 2 ............"
  model_para = pickle.load(open("0_model_maxout_LC_[150, 480, 350, [5, 50]]/modelfile_9"))
  mweight = model_para["model_weight"]
  print mweight.keys()
  #print model_para["model_weight"].keys()
  model = tokenModel(model_para["neg_label"], model_para["net_size"], embeddings, model_para["dropout"], model_para["model_type"], model_para["model_weight"])
  model.evaluate_ready(False)
  print "evaluate ready   is  done ............"
  select2 = active_sample(model, pooldata, active_learner)


  for i in xrange(500,len(select1),500):
    common_rate = computecommon(select1[:i], select2[:i]) /(0.+i)
    print str([i,common_rate])


  '''

  '''

  score_items = model.score_labels(pooldata)
  print score_items[0][0]
  score_items_axu = model_axu.score_labels(pooldata)
  print score_items_axu[0][0]
  success_1 = 0
  success_2 = 0
  success_3 = 0
  false = 0
  for seq_id,item in enumerate(pooldata):
    for index, label in enumerate(item[1]):
	scores = score_items[seq_id][index]
	if label != neg_label and  scores[neg_label] > 0.9:
		scores_axu = score_items_axu[seq_id][index]
		max_score = max(scores_axu[:-1])
		false += 1
		if max_score > 0.1:
		   success_1 += 1
		if max_score > 0.2:
		   success_2 += 1
		if max_score > 0.3:
		   success_3 += 1

  print "succerss 1  :  "+str(success_1)
  print "succerss 2  :  "+str(success_2)
  print "succerss 3  :  "+str(success_3)
  print "false   :   "+str(false)

  

  '''	

  '''
  score_items = model.score_labels(tdata_neg)

  print "neg ins  number ....."+str(len(score_items))

  i = 1
  for index, ma in enumerate(score_items):
  	assert len(tdata_neg[index][1]) == 2*context_len + 1
  	assert len(ma) == 2*context_len + 1
  	assert tdata_neg[index][1][context_len] == tagdic["NONE"]

  	label = tagdic["NONE"]

  	stat[label].append(ma[context_len][label])

  	i += 1
  	if i < 10:
  		print label
  		print ma[context_len]

  for key in stat:
  	stat[key] = sum(stat[key])/len(stat[key])

  print stat





  print pros_items

  candidates = []
  for seq_id, seq in enumerate(pros_items):
  	for token_id, token in enumerate(seq):
  		if pooldata[seq_id][2][token_id] == 1:
  			continue
  		candidates.append((seq_id, token_id, pros_items[seq_id][token_id]))


  select = active_learner(candidates)
  #print select

  added = []
  for item in select:
  	ins = pooldata[item[0]]
  	newx = ins[0]
  	newy = [-1]*item[1] + [ins[1][item[1]]] + [-1]*(len(ins[1]) - item[1] - 1)
  	assert len(newx) == len(newy)
  	added.append((newx, newy))
  	pooldata[item[0]][2][item[1]] = 1

  print select[0]
  print added[0]

  tdata["active"].extend(added)

  model.train_ready()

  para = {}
  para["padding_id"] = padding_id
  para["batch_size"] = batch_size 
  para["score_dir"] = "model_"+str(1)
  para["token_num"] = token_num
  model.train(tdata, devdata, testdata, para)
  '''





