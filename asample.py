import sys
import pickle
from token_model import *
import numpy as np
from util import *
def baseline_random(query_num):
	print "adopt  random  sampling ..... ....... "
	def random_select(data):
		for i in xrange(len(data)-1):
			randindex = random.randint(i+1, len(data) - 1)

			data[i], data[randindex] = data[randindex], data[i]

		return data[:query_num]

	return random_select


def learner_LC(query_num): #least confidence

	def active_learner(data):
		# each item of data is (sequence id, token id, pro distribution)

		candidates = []
		for ins in data:
			pros = ins[2]
			max_pro = max(pros)
			candidates.append((ins[0], ins[1], max_pro))

		candidates.sort(lambda x,y: cmp(x[2],y[2]))

		return candidates[:query_num]

	return active_learner

def learner_margin(query_num): # margin ( max - 2 st max)
	
	def active_learner(data):
		# each item of data is (sequence id, token id, pro distribution)
		candidates = []
		for ins in data:
			pros = ins[2]
			sorted_pros = np.sort(pros)[::-1]
			margin =  sorted_pros[0] - sorted_pros[1]

			candidates.append((ins[0], ins[1], margin))

		candidates.sort(lambda x,y: cmp(x[2],y[2]))

		return candidates[:query_num]

	return active_learner

def  learner_entropy(query_num): # entropy

	def active_learner(data):
		candidates = []
		for ins in data:
			pros = ins[2]
			entropy  = sum(0 - pro*np.log(pro) for pro in pros )
			candidates.append((ins[0],ins[1], entropy))


		candidates.sort(lambda x,y : cmp(y[2], x[2]))

		return candidates[:query_num]

	return active_learner

def learner_SVE(query_num): #vote entropy
	
	def active_learner(data, weights = None):
		candidates = []
		for ins in data:
			multiple_pros = ins[2]# a list of array, each array is a probability distribution 
			if weights is None:
				aver_pro = sum(multiple_pros) / len(multiple_pros)
			else:
				assert len(weights) == len(multiple_pros)
				aver_pro = sum(pros*weight for pros, weight in zip(multiple_pros, weights))


			vote_entropy = sum(0 - pro*np.log(pro) for pro in aver_pro)
			candidates.append((ins[0], ins[1], vote_entropy))

		candidates.sort(lambda x,y : cmp(y[2], x[2]))

		return candidates[:query_num]
	return active_learner


def baseline_active_sample(model, pooldata, active_learner,vote_num = 0, isvote = False):
  if isvote:
  	# adopt original drop out value to approximate mutiple models
  	print "adopt  vote entropy active  sample........."
  	assert vote_num > 1
  	pros_list = []
  	for i in xrange(vote_num):
  		pros_list.append(model.score_labels(pooldata))
  	candidates = []
  	pros_items = pros_list[0]
  	for seq_id, seq in enumerate(pros_items):
		for token_id, token in enumerate(seq):
			if pooldata[seq_id][2][token_id] == 1:
				continue
			scores = []
			for index in xrange(vote_num):
				scores.append(pros_list[index][seq_id][token_id])
			candidates.append((seq_id, token_id, scores))
	select = active_learner(candidates)

  
  else:
  	print "adopt  uncertainty based active sample ........"
  	current = model.get_droppro()
  	print "previous model dropout  ....  "+str(model.get_droppro())
	model.set_droppro(0)
	print "current model dropout .... "+str(model.get_droppro())

  	pros_items = model.score_labels(pooldata) # a list of probability distributions, each instance  is a seqeunce of pro dis, each pro dis responds to a token
  	#pros_items = [item[0] for item in pooldata]
  	model.set_droppro(current)
	candidates = []
	for seq_id, seq in enumerate(pros_items):
		for token_id, token in enumerate(seq):
			if pooldata[seq_id][2][token_id] == 1:
				continue
			candidates.append((seq_id, token_id, pros_items[seq_id][token_id]))


	select = active_learner(candidates)
  
  #debug
  #print select[:10]
  #print "........................"


  for item in select:
    pooldata[item[0]][2][item[1]] = 1

  return select

def randompos_active_sample(model, pooldata, active_learner,neg_label = 5):


	print "adopt  positives  random   sample ........"
	current = model.get_droppro()
	print "previous model dropout  ....  "+str(model.get_droppro())
	model.set_droppro(0)
	print "current model dropout .... "+str(model.get_droppro())

	pros_items = model.score_labels(pooldata) # a list of probability distributions, each instance  is a seqeunce of pro dis, each pro dis responds to a token
	model.set_droppro(current)
	candidates = []
	pos_cans = []
	for seq_id, seq in enumerate(pros_items):
		for token_id, token in enumerate(seq):
			if pooldata[seq_id][2][token_id] == 1:
				continue
			if pooldata[seq_id][1][token_id] != neg_label:
				pos_cans.append((seq_id,token_id))

			candidates.append((seq_id, token_id, pros_items[seq_id][token_id]))


	select = active_learner(candidates)

    
	negnum  = 0
	for item in select:
		if pooldata[item[0]][1][item[1]] == neg_label:
			pooldata[item[0]][2][item[1]] = 1
			negnum += 1
	
	randomdata(pos_cans)

	posnum = len(select) - negnum
	pos_select = pos_cans[:posnum]

	for item in pos_select:
		pooldata[item[0]][2][item[1]] = 1

	print "select  pos  num ........ ... "+str(posnum)
	print "select  neg  num  ............"+str(negnum)

	return select

def random_active_sample(pos,neg, pooldata, neg_label = 5, israndom_neg = True,model = None, active_learner = None):

	print "adopt positive  random  sample ..... ..... . "

	if not israndom_neg:
		print "negtives sample  adopts active learning...................."
		current = model.get_droppro()
		print "previous model dropout  ....  "+str(model.get_droppro())
		model.set_droppro(0)
		print "current model dropout .... "+str(model.get_droppro())

		pros_items = model.score_labels(pooldata) # a list of probability distributions, each instance  is a seqeunce of pro dis, each pro dis responds to a token
		model.set_droppro(current)
		
	else:
		print "negatives sample  adopts random sampling ..............."
	pos_cans = []
	neg_cans = []
	for seq_id, seq in enumerate(pooldata):
	
		for token_id, token in enumerate(seq[0]):
			if pooldata[seq_id][2][token_id] == 1:
				continue
			if pooldata[seq_id][1][token_id] != neg_label:
				pos_cans.append((seq_id,token_id))
			else:
				if israndom_neg:
					neg_cans.append((seq_id,token_id))
				else:
					neg_cans.append((seq_id,token_id, pros_items[seq_id][token_id]))



	if israndom_neg:
		randomdata(neg_cans)
		neg_select = neg_cans[:neg]
	else:
		neg_select = active_learner(neg_cans)


	randomdata(pos_cans)
	pos_select = pos_cans[:pos]


    
	for item in neg_select+pos_select:
		pooldata[item[0]][2][item[1]] = 1
	
	

	print "select  pos  num ........ ... "+str(len(pos_select))
	print "select  neg  num  ............"+str(len(neg_select))

	return pos_select + neg_select

def my_active_sample(models, weights,pooldata,active_learner):
	pros_list = []
	for model in models:
		model.set_droppro(0.)
		pros_list.append(model.score_labels(pooldata))
  	
  	candidates = []
  	pros_items = pros_list[0]
  	for seq_id, seq in enumerate(pros_items):
		for token_id, token in enumerate(seq):
			if pooldata[seq_id][2][token_id] == 1:
				continue
			scores = []
			for index in xrange(len(models)):
				scores.append(pros_list[index][seq_id][token_id])
			assert len(weights) == len(scores)
			ave_pro = sum(pro*weight for pro, weight in zip(scores, weights))
			candidates.append((seq_id, token_id, ave_pro))
	
	select = active_learner(candidates)

	#debug
	'''
  	print select[:4]
  	for item in select[:4]:
  		for pros in pros_list:
  			print pros[item[0]][item[1]]
  	print "........................"
  	'''

  	for item in select:
  		pooldata[item[0]][2][item[1]] = 1

  	return select


def axu_active_sample(model,model_axu, threshold, pooldata,active_learner):
	
	model.set_droppro(0.)
	result = model.score_labels(pooldata)
	
	model_axu.set_droppro(0.)
	result_axu = model_axu.score_labels(pooldata)
  	
  	candidates = []
  	pros_items = result
  	for seq_id, seq in enumerate(pros_items):
		for token_id, token in enumerate(seq):
			if pooldata[seq_id][2][token_id] == 1:
				continue
			print token
			if token[5] > threshold:
				candidates.append((seq_id, token_id, result_axu[seq_id][token_id]))
			else:
				candidates.append((seq_id, token_id, token))
	
	select = active_learner(candidates)

	#debug
	'''
  	print select[:4]
  	for item in select[:4]:
  		for pros in pros_list:
  			print pros[item[0]][item[1]]
  	print "........................"
  	'''

  	for item in select:
  		pooldata[item[0]][2][item[1]] = 1

  	return select


def analyze(model, pooldata, threshold):
	model.set_droppro(0.)
	result = model.score_labels(pooldata)
	
	pos_sum = 0
	select = []
  	pros_items = result
  	for seq_id, seq in enumerate(pros_items):
		for token_id, token in enumerate(seq):
			if pooldata[seq_id][2][token_id] == 1:
				continue
			if pooldata[seq_id][1][token_id] != 5:
				pos_sum += 1
				if  token[5] > threshold:
					select.append((seq_id, token_id, token))
	'''
	randomdata(select)
	select = select[:2000]
  	for item in select:
  		pooldata[item[0]][2][item[1]] = 1
  	'''
	print "miss num  ............. "+str(len(select))
	print "all  pos in pool ......."+str(pos_sum)
	

  	return select


def computecommon(select1, select2):
  # compute common select  between two select
  s = set([str(item[0])+"_"+str(item[1]) for item in select1])
  stat = 0
  for item in select2: 
    key = str(item[0])+"_"+str(item[1])
    if key in s:
      stat += 1

  return stat


if __name__ == "__main__":
  
  active_type = sys.argv[1]
  sample_num = int(sys.argv[2])
  pool_file = sys.argv[3]
  newpool_file = sys.argv[4]
 

  if active_type == "LC":
    active_learner = learner_LC(sample_num)
    print "adopt  least confidence  active  learner ........"
  elif active_type == "margin":
    active_learner = learner_margin(sample_num)
    print "adopt  margin  based  active  learner  ........"
  elif active_type == "entropy":
    active_learner = learner_entropy(sample_num)
    print  "adopt  entropy  based   active  learner ........"
  elif active_type == "vote_entropy":
  	active_learner = learner_SVE(sample_num)
  	print "adopt  vote entropy  active learner ........."
  elif active_type == "random":
    active_learner = baseline_random(sample_num)
    print "random  sampling .............."

  print "load  pool  data  ............"+pool_file
  pooldata = pickle.load(open(pool_file))

  
  handpool = pickle.load(open("tradition_baseline/wholedata/4000_active_data/new_round40008/pooldata"))
  handpool2ins(pooldata, 0, 2,handpool["pool_mark"])
  

  select = pool2loc(pooldata)
  print  "pooldata slecet num  :  "+str(len(select))
  embeddings= pickle.load(open("charembeddings"))

   
  
  model_file = sys.argv[5]
  #model_file = "softmax_baseline/modelfile_6"
  print "loading  model ............... "+model_file
  model_para = pickle.load(open(model_file))
  model = tokenModel(model_para["neg_label"], model_para["net_size"], "LSTM_forward", embeddings, drop_pro = model_para["dropout"], model_type = model_para["model_type"], premodel = model_para["model_weight"])
  model.evaluate_ready()
  #select = baseline_active_sample(model, pooldata, active_learner, isvote = False )
  select = analyze(model, pooldata, 0.8 )
  #print "baseline  miss  num  :  "+str(len(select))   
  #print select[:10]
  pickle.dump(pooldata, open(newpool_file, "w+"))
  
  '''
  model_file = sys.argv[6]
  print "loading  model ............... "+model_file
  model_para = pickle.load(open(model_file))
  model2 = tokenModel(model_para["neg_label"], model_para["net_size"], "LSTM_LSTM", embeddings, model_para["dropout"], model_para["model_type"], model_para["model_weight"])
  model2.evaluate_ready()
  
  print "evaluate ready   is  done ............"
  #select  = my_active_sample([model, model2], [0.5,0.5], pooldata,active_learner)
  select2 = analyze(model2, pooldata, 0.9 )
  myselect = computecommon(select, select2)
  print "my  method  miss num   :   "+str(myselect)

  pickle.dump(pooldata, open(newpool_file, "w+"))
  '''

