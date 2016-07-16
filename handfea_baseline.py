# encoding=utf-8
from weiboNER_features import *
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model  import LogisticRegression
import pickle
import sys
import numpy as np
import os
import shutil
import scipy
from util import *

def gen_feature_data(rawdata):

	fea_data = [feature_extractor(readiter(zip([str(charid) for charid in item[0]], item[1]), ['w','L'])) for item in rawdata] 
	return fea_data



def gen_cinput(origindata, pooldata = [],threshold = 5):
	origin_feas = gen_feature_data(origindata)
	pool_feas = gen_feature_data(pooldata)

	feas_X = []
	label_Y = []
	s  = set()
	for seq in origin_feas:
		feas_X.extend([item["F"] for item in seq])
		for item in seq:
			s.update(item["F"])
		label_Y.extend([item["L"] for item in seq])

	assert len(feas_X) == len(label_Y)
	print "original  data  data  num   :   "+str(len(feas_X))

	
	feas_X_2 = []
	label_Y_2 = []
	for seq_id, seq in enumerate(pool_feas):
		for token_id, token in enumerate(seq):
			if pooldata[seq_id][2][token_id] == 1:
				feas_X_2.append(token["F"])
				s.update(token["F"])
				label_Y_2.append(token["L"])



	print "pool data  data  num   :   "+str(len(feas_X_2))


	print "original feature num   ................ "+str(len(s))

	X = feas_X + feas_X_2
	X = featurefilter(X, threshold)
	print X[:2]


	Y = label_Y + label_Y_2
	h = FeatureHasher(input_type = "string", non_negative = True)

	X = h.transform(X)

	return X ,Y, h 


def featurefilter(X, threshold):
	d = {}
	for item in X:
		for fea in item:
			d.setdefault(fea,0)
			d[fea] += 1

	pairs = d.items()

	s = set()
	for pair in pairs:
		if pair[1] >= threshold:
			s.add(pair[0])

	print "select  feature num  ........................  "+str(len(s))

	newx = []
	for item in X:
		newx.append([fea for fea in item if fea in s])


	return newx



def compute_pr(predict, gold, neg_label):
	overall = 	sum(1 for item in gold if item != neg_label)
	correct = sum(1 for index, item in enumerate(predict) if item != neg_label and item == gold[index])
	extract = sum(1 for item in predict if item != neg_label)

	precision = correct / (extract+1e-7)
	recall = correct /  (overall+1e-7)
	f = 2./(1./(precision+1e-7)+1./(recall+1e-7))


	return precision, recall, f

def active_sample(classifier, pool_X, pool_Y, pool_mark, active_type, sample_num,israndom = False):
	pros_list = classifier.predict_proba(pool_X)

	print "if adopt  random  positive  sampling  ????????"+str(israndom)

	if active_type == "LC":

		stat = 0
		candidates = []
		for index, pros in enumerate(pros_list):
			if pool_mark[index] == 1:
				stat += 1
				continue
			candidates.append((max(pros), index))


		print "pooldata already  selected data num  ..... ......  "+str(stat)
		candidates.sort(lambda x,y: cmp(x[0], y[0]))
		print candidates[:5]

		select = candidates[:sample_num]

		added_X = []
		added_Y = []
		for item in select:
			if israndom:
				if pool_Y[item[1]] == 5:
					pool_mark[item[1]] = 1
					added_X.append(pool_X[item[1]])
					added_Y.append(pool_Y[item[1]])


			else:
				pool_mark[item[1]] = 1
				added_X.append(pool_X[item[1]])
				added_Y.append(pool_Y[item[1]])

		if israndom:
			neg_num = len(added_Y)
			print "select  negtive  data  num  ..... .....  "+str(neg_num)
			pos_num = sample_num - neg_num
			print "select  positive  data  num  ... .....  " +str(pos_num)
			pos_cans = []
			for item in candidates:
				assert pool_Y[item[1]] < 6 
				if pool_Y[item[1]] != 5:
					pos_cans.append(item)

			randomdata(pos_cans)
			pos_select = pos_cans[:pos_num]
			for item in pos_select:
				pool_mark[item[1]] = 1
				added_X.append(pool_X[item[1]])
				added_Y.append(pool_Y[item[1]])

		assert len(added_Y) == sample_num
		return added_X, added_Y

	else:
		raise Exception


def random_sample(pool_X, pool_Y, pool_mark,pos_sample_num,neg_sample_num):



	stat = 0
	pos_cans = []
	neg_cans = []
	for index, mark in enumerate(pool_mark):
		if mark == 1:
			stat += 1
			continue

		assert pool_Y[index] < 6
		if pool_Y[index] == 5:
			neg_cans.append(index)
		else:
			pos_cans.append(index)


	print "pooldata already  selected data num  ..... ......  "+str(stat)
	
	randomdata(pos_cans)
	randomdata(neg_cans)

	candidates = pos_cans[:pos_sample_num] + neg_cans[:neg_sample_num]

	added_X = []
	added_Y = []
	for item in candidates:
		pool_mark[item] = 1
		added_X.append(pool_X[item])
		added_Y.append(pool_Y[item])

	return added_X, added_Y


def addpool(X,Y, pool_X,pool_Y,pool_mark):
	newx = [] 
	newy = []
	for index, mark in enumerate(pool_mark):
		if mark == 1:
			newx.append(pool_X[index])
			newy.append(pool_Y[index])


	train_X = scipy.sparse.vstack([X]+newx)
	train_Y = Y + newy

	return train_X, train_Y






if __name__ == "__main__":

	active_type = sys.argv[1]
	sample_num = int(sys.argv[2])

	traindata = pickle.load(open("cleandata/train17k/traindata"))
	devdata = pickle.load(open("cleandata/train17k/devdata"))
	testdata = pickle.load(open("cleandata/train17k/testdata"))
	#pooldata = pickle.load(open("cleandata/pooldata"))
	pooldata = pickle.load(open(sys.argv[4]))
	print "training  data   num  ................. "+str(len(traindata)) + "   dev  data  num ................  "+str(len(devdata))+"   test  data  num  ................  "+str(len(testdata))

	#print traindata[:2]

	neg_label	 = 5
	


	point1 = sum(len(item[0]) for item in traindata)
	point2 = sum(len(item[0]) for item in traindata + devdata)
	point3 = sum(len(item[0]) for item in traindata + devdata + testdata)
	point4 = sum(len(item[0]) for item in traindata + devdata + testdata + pooldata)

	X, Y, _  = gen_cinput(traindata+devdata+testdata+pooldata, pooldata, threshold = 5)
	#print X[:5]
	#print Y[:5]
	train_X,train_Y = X[:point1], Y[:point1] 
	dev_X, dev_Y,  = X[point1: point2], Y[point1: point2]
	test_X, test_Y = X[point2:point3], Y[point2:point3]
	pool_X, pool_Y = X[point3:point4], Y[point3:point4]

	pool_mark = [0]*len(pool_Y)
	


	'''
	pdata = pickle.load(open(sys.argv[5]+"/pooldata"))
	pool_X = pdata["pool_X"]
	pool_Y = pdata["pool_Y"]
	pool_mark = pdata["pool_mark"]

	train_X, train_Y = addpool(train_X,train_Y,pool_X,pool_Y,pool_mark)
	'''

	#random sample
	#added_X, added_Y = random_sample(pool_X, pool_Y, pool_mark,1000, 3000)
	#print "random  data num  .......  ..   "+str(len(added_Y))
	#train_X = scipy.sparse.vstack([train_X]+added_X)
	#train_Y += added_Y




	#upbound 
	#train_X = scipy.sparse.vstack([train_X, pool_X])
	#train_Y += pool_Y


	#add  select data
	#added_X, added_Y = X[point4:], Y[point4:]
	#train_X = scipy.sparse.vstack([train_X, added_X])
	#train_Y += added_Y



	


	for around in xrange(14):
		score_dir = sys.argv[3]+str(around)
		if os.path.exists(score_dir):
			shutil.rmtree(score_dir)

		os.mkdir(score_dir)

		pdata = {}
		pdata["pool_X"] = pool_X
		pdata["pool_Y"] = pool_Y
		pdata["pool_mark"] = pool_mark

		pickle.dump(pdata, open(score_dir+"/pooldata", "w+"))
		classifier = LogisticRegression(penalty = 'l2', C = 2, solver = 'lbfgs',max_iter = 10, multi_class='multinomial', warm_start = True)


		print "training  data   num  ................. "+str(len(train_Y)) + "   dev  data  num ................  "+str(len(dev_Y))+"   test  data  num  ................  "+str(len(test_Y))


		for epoch in xrange(10):
			classifier.fit(train_X,train_Y)


			pre_Y = classifier.predict(train_X)
			#print pre_Y[:50]
			precision, recall, f = compute_pr(pre_Y, train_Y, neg_label)
			s = "train   score  ........  epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
			print s
			trainscore_file = open(score_dir+"/trainscore",'ab')
			trainscore_file.write(s)
			trainscore_file.close()



			pre_Y = classifier.predict(dev_X)
			#print pre_Y[:50]
			precision, recall, f = compute_pr(pre_Y, dev_Y, neg_label)
			s = "dev  score  ........  epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
			print s
			trainscore_file = open(score_dir+"/devscore",'ab')
			trainscore_file.write(s)
			trainscore_file.close()



			pre_Y = classifier.predict(test_X)
			#print pre_Y[:50]
			precision, recall, f = compute_pr(pre_Y, test_Y, neg_label)
			s = "test  sco re  ........  epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
			print s
			trainscore_file = open(score_dir+"/testscore",'ab')
			trainscore_file.write(s)
			trainscore_file.close()

		
		added_X, added_Y = active_sample(classifier, pool_X, pool_Y, pool_mark, active_type, sample_num, israndom = False)

		train_X = scipy.sparse.vstack([train_X]+added_X)
		train_Y += added_Y










	

