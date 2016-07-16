import numpy as np
from BLSTM import model
import io
def learner_LC(query_num): #least confidence

	def active_learner(model, data):
		candidates = []
		for index, ins in enumerate(data):
			result = model.decode(ins, 1)
			candidates.append((index, result[0][1]))

		candidates.sort(lambda x,y: cmp(x[1],y[1]))

		return [item[0] for item in candidates[:query_num]]

	return active_learner

def learner_margin(query_num): # margin ( max - 2 st max)
	
	def active_learner(model, data):
		candidates = []
		for index, ins in enumerate(data):
			result = model.decode(ins, 2)
			candidates.append((index, np.exp(result[0][1]) - np.exp(result[1][1])))

		candidates.sort(lambda x,y: cmp(x[1],y[1]))

		return [item[0] for item in candidates[:query_num]]

	return active_learner

def  learner_NSE(query_num, N): # N-best sequence entropy

	def active_learner(model, data):
		candidates = []

		for index, ins  in enumerate(data):
			result = model.beamsearch(ins, 2*N, N)
			entropy  = sum(0 - np.exp(item[1])*item[1] for item in result )
			candidates.append(index, entropy)


		candidates.sort(lambda x,y : cmp(y[1], x[1]))

		return [item[0] for item in candidates[:query_num]]

	return active_learner

def learner_SVE(query_num, N): #N-best sequence vote entropy
	
	def active_learner(models, data):
		candidates = []
		for index, ins in enumerate(data):
			seq_pro = {}
			for model in models:
				result = model.beamsearch(ins, 4*N, 2*N)
				for item in result:
					str_seq  =  "".join(str(tag) for tag in item[0])
					seq_pro.setdefault(str_seq, 0)
					seq_pro[str_seq] += np.exp(item[1])
			seq_pro = [item / len(models) for item in seq_pro.values()]

			seq_pro.sort(reverse = True)

			vote_entropy = sum(0 - pro*np.lop(pro) for pro in seq_pro[:N])
			candidates.append(index, vote_entropy)

		candidates.sort(lambda x,y : cmp(y[1], x[1]))

		return [item[0] for item in candidates[:query_num]]
	return active_learner


if __name__ == "__main__":
	

	dim = 128
	em_num = 1
	iterate = 5
	active_learner = ""

	print "loading model ......................."
	embeddic = pickle.load(open(""))
	tokendic = pickle.load(open(""))
	padding_id = wdic["<padding>"]
	pids = [padding_id]
	loadedmodel = pickle.load(open(""))
	tags = pickle.load(open(""))
	model_size = pickle.load(open(""))
	


	traindata =  io.read_weibodata("", tokendic, tags)
	devdata =  io.read_weibodata("", tokendic, tags)
	testdata =  io.read_weibodata("", tokendic, tags)
	pooldata =  io.read_weibodata("", tokendic, tags)

	print "model  initialization ..............network size  "+str(model_size)
	model = model(len(tags),em_num, model_size,dropout = 0, embeddic = embeddic, premodel = loadedmodel)
    print "model evaluating preparing.........................."
    model.evaluate_ready()

    for i in xrange(iterate):

    	
    	pool_sen = [item[0] for item in pooldata]
    	indexlist = active_learner(model, pool_sen)
    	indexlist.sort(reverse = True)
    	
    	for index in indexlist:
    		traindata.append(pooldata[index])
    		del pooldata[index]

    	
    	#add new data  and re-train model  

    	epoch = 0
    	while True:
    	
	    	supergroups = preprocess.data2batch(traindata,batch_size,pids)
	    	print "iterate ................"+str(i)
	        print "start  supervision update .............."
	        for sgroup in supergroups:
	            #model.train_ready(group)
	            sgroupscore = model.upAndEva(sgroup)
	            print "group loss : "+str(sgroupscore)


	        print "update over  ....."


	        predict = []
	        gold = []
	        print  "start  decoding   .............."
	        for item in devdata:
	        	predict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
	            gold.append(item[1])
	        print  "decoding   finished  ..........."
	        precision, recall = pr.computePR(tags, gold, predict)
	        f = 2./(1./(precision+1e-7)+1./(recall+1e-7))
	        score_file = open(modelstr+"/devscore",'ab')
	        print  "develop  result   :   "
	        s = "iterate  :  "+str(i)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
	        print s
	        score_file.write(s)

	        epoch += 1













	        





