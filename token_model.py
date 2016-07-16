# encoding=utf-8
import theano.tensor as T
import theano
import numpy as np
import pickle
from NNmodel import NNmodel
import random
import os
import shutil
#from token_alearner import resetpool, active_sample

def shared32(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)
def resetpool(pooldata):
	for ins in pooldata:
		for loc in xrange(len(ins[2])):
			ins[2][loc] = 0


def pool2ins(pooldata,padding_id, context_len):
# generate training instances from marked pooldata
	added = []
  	for ins in pooldata:
  		for loc, token in enumerate(ins[2]):
  			if token == 1:
  				ins_x = [padding_id]*context_len + ins[0]+ [padding_id]*context_len
  				newx = ins_x[loc: loc + 2*context_len + 1]
  				newy = [-1]*context_len + [ins[1][loc]] + [-1]*context_len
  				assert len(newx) == len(newy)
  				added.append((newx, newy))

  	return added



def handpool2ins(pooldata,padding_id, context_len,handpool_mark):
# generate training instances from marked pooldata
	added = []
	index = 0
  	for ins in pooldata:
  		for loc, token in enumerate(ins[2]):
  			if handpool_mark[index] == 1:
  				ins[2][loc] = 1
  				ins_x = [padding_id]*context_len + ins[0]+ [padding_id]*context_len
  				newx = ins_x[loc: loc + 2*context_len + 1]
  				newy = [-1]*context_len + [ins[1][loc]] + [-1]*context_len
  				assert len(newx) == len(newy)
  				added.append((newx, newy))
  			index += 1

  	return added 
def pool2loc(pooldata):

	added = []
  	for seq_id,ins in enumerate(pooldata):
  		for loc, token in enumerate(ins[2]):
  			if token == 1:
  				added.append((str(seq_id)+"_"+str(loc), ins[1][loc]))
	return added


def active_sample(model, pooldata, active_learner):
	pros_items = model.score_labels(pooldata) # a list of probability distributions, each instance  is a seqeunce of pro dis, each pro dis responds to a token
  	candidates = []
  	for seq_id, seq in enumerate(pros_items):
  		for token_id, token in enumerate(seq):
  			if pooldata[seq_id][2][token_id] == 1:
  				continue
  			candidates.append((seq_id, token_id, pros_items[seq_id][token_id]))

  	select = active_learner(candidates)
  	for item in select:
  		pooldata[item[0]][2][item[1]] = 1




class tokenModel(NNmodel):
	def __init__(self,neg_label, net_size,midstr,embeddic, mid_id = 0,conreg_weight1 = 0,conreg_weight2 = 0,variance = 1, nc =1, drop_pro = 0, model_type = "softmax", premodel = {},  fake_label = -1, 
		pos_ratio = 0.5, mweight = 0.9, lrate = 0.1, opt = "momentum", wbound = 1, gradbound = 15, wdecay = 0.,mo_wdecay = 0, fix_emb = True,ismention = False,istransition = False):
		NNmodel.__init__(self,wdecay,opt,wbound, gradbound,mweight,lrate)
		def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
		def f_rectlin(x): return x*(x>0)
		def f_rectlin2(x): return x*(x>0) + 0.01 * x

		self.nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2}


		self.model_type = model_type
		self.tag_num = net_size[-1]
		self.ismention =ismention
		self.istransition = istransition

		if model_type == "maxneg" or model_type == "maxout":
			assert len(net_size[-1]) == 2
		self.midstr = midstr
		self.net_size = net_size
		self.fake_label = fake_label
		self.neg_label = neg_label
		self.fix_emb = fix_emb
		self.mo_wdecay = mo_wdecay
		self.pos_ratio = pos_ratio
		self.mid_id = mid_id
		self.conreg_weight1 = conreg_weight1
		self.conreg_weight2 = conreg_weight2
		self.variance = variance
		self.nc = nc
		
		print "whether  fixing  word embedding  :  "+str(self.fix_emb)

		self.w = premodel
		self.dic = shared32(embeddic)

		self.dropout_prob = shared32(drop_pro)
		self.srng = T.shared_randomstreams.RandomStreams(random.randint(0,9999))

	def train_ready(self):
		var_x = T.imatrix()
		var_y = T.imatrix()

		#debug
		#var_x.tag.test_value = np.asarray([[1,2,3],[2,3,4]], dtype = 'int32')
		#var_y.tag.test_value = np.asarray([[2,5,5],[5,5,5]], dtype = 'int32')
		if self.istransition:
			print "add  transition matrix .................... ....................... trans "
			self.w["trans"] = shared32(np.random.randn(self.net_size[-1]+1,self.net_size[-1]))

		if self.model_type == "softmax":
			print "adopt softmax model..........."
			loss = self.l2reg(self.w, self.wdecay)+self.logp_loss1(var_x,var_y,self.fake_label)

			witems = self.w.values()
			#ave_w = sum(T.sum(item**2) for item in witems)/len(witems)
			wg = T.grad(loss, witems)
			#ave_g = sum(T.sum(item**2) for item in wg) /len(wg)

			weight_up = self.upda(wg, witems, self.lrate, self.mweight, self.opt, self.gradbound)

			if not self.fix_emb:
				dicitems = self.dic.values()
				dg = T.grad(loss, dicitems)

				dic_up = self.upda(dg, dicitems, self.lrate/10., self.mweight, self.opt)
				weight_up.update(dic_up)

			up  = weight_up

			self.updatefunc = theano.function([var_x, var_y], loss,updates = up)
		elif self.model_type == "softmax_reg":
			print "adopt softmax model plus contractive regularization ........ "
			print "weight 1   : "+str(self.conreg_weight1)
			print "weight 2   : "+str(self.conreg_weight2)
			print "variance   : "+str(self.variance)
			print "nc  : "+str(self.nc)

			loss = self.logp_reg(var_x,var_y,self.fake_label, self.conreg_weight1, self.conreg_weight2, self.variance,self.nc)
		

			witems = self.w.values()
			#ave_w = sum(T.sum(item**2) for item in witems)/len(witems)
			wg = T.grad(loss, witems)
			#ave_g = sum(T.sum(item**2) for item in wg) /len(wg)

			weight_up = self.upda(wg, witems, self.lrate, self.mweight, self.opt, self.gradbound)

			if not self.fix_emb:
				dicitems = self.dic.values()
				dg = T.grad(loss, dicitems)

				dic_up = self.upda(dg, dicitems, self.lrate/10., self.mweight, self.opt)
				weight_up.update(dic_up)

			up  = weight_up

			self.updatefunc = theano.function([var_x, var_y], loss,updates = up)


		elif self.model_type == "maxneg":
			print "adopt softmax and max negative model .........."
			loss1 = self.l2reg(self.w, self.wdecay)+self.logp_loss2(var_x,var_y,self.fake_label,self.neg_label,False)# normal classification loss
			loss2 = self.l2reg(self.w, self.wdecay)+self.logp_loss2(var_x,var_y,self.fake_label,self.neg_label,True)# max neg  loss


			witems = self.w.values()
			#ave_w = sum(T.sum(item**2) for item in witems)/len(witems)
			
			wg1 = T.grad(loss1, witems)
			#ave_g = sum(T.sum(item**2) for item in wg) /len(wg)

			weight_up = self.upda(wg1, witems, self.lrate, self.mweight, self.opt, self.gradbound)

			if not self.fix_emb:
				dicitems = [self.dic]
				dg1 = T.grad(loss1, dicitems)
				dic_up = self.upda(dg1, dicitems, self.lrate/10., self.mweight, self.opt)
				weight_up.update(dic_up)

			up  = weight_up

			self.updatefunc1 = theano.function([var_x, var_y], loss1,updates = up)

			wg2 = T.grad(loss2, witems)

			weight_up = self.upda(wg2, witems, self.lrate, self.mweight, self.opt, self.gradbound)

			if not self.fix_emb:
				dicitems = [self.dic]
				dg2 = T.grad(loss2, dicitems)
				dic_up = self.upda(dg2, dicitems, self.lrate/10., self.mweight, self.opt)
				weight_up.update(dic_up)

			up = weight_up

			self.updatefunc2 = theano.function([var_x, var_y], loss2,updates = up)

		elif self.model_type == "maxout":
			print "adopt maxout model............."
			allloss, pos_loss, neg_loss = self.logp_loss3(var_x,var_y,self.fake_label,self.neg_label,self.pos_ratio)# neg   maxout   loss
			loss = self.l2reg(self.w, self.wdecay)+ allloss + self.regmaxw(self.mo_wdecay)
			
			maxw = self.w.pop("maxw")
			witems = self.w.values()
			wg = T.grad(loss, witems)
			weight_up = self.upda(wg, witems, self.lrate, self.mweight, self.opt, self.gradbound)

			maxwg = T.grad(loss, [maxw])
			max_up = self.upda(maxwg, [maxw], self.lrate, self.mweight, self.opt, wbound = self.wbound)
			
			weight_up.update(max_up)
			self.w["maxw"] = maxw

			if not self.fix_emb:
				dicitems = self.dic.values()
				dg = T.grad(loss, dicitems)

				dic_up = self.upda(dg, dicitems, self.lrate/10., self.mweight, self.opt)
				weight_up.update(dic_up)

			up  = weight_up

			self.updatefunc = theano.function([var_x, var_y], [loss, pos_loss, neg_loss,T.sum(maxw**2, 0)],updates = up)
		else: raise Exception






	def evaluate_ready(self, ispro = True):
		var_x = T.ivector()
		var_y = T.ivector()

		print "adopt   mention  level  evaluate ????????????????????       "+str(self.ismention)
		if self.model_type == "softmax" or self.model_type == "softmax_reg":

			if self.istransition:
				output = self.structure1(var_x, ispro = False)
				self.evafunc = theano.function([var_x], output)

			else:
				output = self.structure1(var_x, ispro)
				self.evafunc = theano.function([var_x], output)

		
		elif self.model_type == "maxneg":
			out1, out2 = self.structure2(var_x,ispro)
			self.evafunc = theano.function([var_x], [out1,out2])

		elif self.model_type == "maxout":

			out1, out2 = self.structure2(var_x,False)
			self.evafunc = theano.function([var_x], [out1,out2])
		else: raise Exception





	def train(self, tdata, traindata=None, devdata=None, pooldata=None, para=None, testdata=None , turnpoint = -1,active_learner = None):

		
		batch_size = para["batch_size"]
		score_dir = para["score_dir"]
		lrates = para["lrates"]
		posnum = para["posnum"]
		negnum = para["negnum"]

		if os.path.exists(score_dir):
			shutil.rmtree(score_dir)
		os.mkdir(score_dir)
		bestf = 0
		pickle.dump(pooldata, open(score_dir+"/pooldata_best", 'w+'))
		pickle.dump(lrates, open(score_dir+"/lrates", "w+"))

		for epoch, lrate in enumerate(lrates):

			self.set_lrate(lrate)

		
				


			print "new epoch start....................."+str(epoch)
			print "current learning rate .............."+str(lrate)
			'''
			if epoch % 2 == 0:
				allbatches = self.create_batch2(tdata,  batch_size)
				batches = allbatches[:len(allbatches)/2]

			else:
				batches = allbatches[len(allbatches)/2:]
			'''

			batches = self.create_batch2(tdata,  batch_size)

			negsum, possum = 0., 0.
			for batch in batches:
				#loss, ave_w, ave_g = self.updatefunc(batch[0], batch[1])
				if self.model_type == "maxout":
					loss, posloss, negloss, maxwnorm = self.updatefunc(batch[0], batch[1])
					possum += posloss
					negsum += negloss
					print "maxout weight l2 norm..........."+str(maxwnorm) 
				else:
					loss = self.updatefunc(batch[0], batch[1])
				print "batch loss  :  "+str(loss)
				#print "average weight : "+str(ave_w)
				#print "average gradient : "+str(ave_g)
			print "pos exmaple average  loss : "+str(possum / posnum )
			print "neg exmaple average  loss : "+str(negsum / negnum )

			
			if self.model_type == "maxneg" or self.model_type == "maxout":
				if epoch > 1 and epoch % 5 == 0:
					resetpool(pooldata)
					print "start active  sample..............."
					active_sample(self, pooldata, active_learner)

					print "write  select  data.............."
					pickle.dump(pooldata, open(score_dir+"/pooldata_epoch"+str(epoch), "w+"))

					print "write  current model ...................."
					modelfile = score_dir+"/modelfile_"+str(epoch)
					self.dumpmodel(modelfile)
					resetpool(pooldata)
			else:
				if epoch > -1:
					modelfile = score_dir+"/modelfile_"+str(epoch)
					self.dumpmodel(modelfile)


			# test on train data
			con_tdata = traindata
			if len(traindata) > 2000:
				con_tdata = traindata[:2000]
			precision,recall,f = self.evaluate(con_tdata, self.neg_label,self.ismention)
			s = "train   score  ........  epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"

			print s

			trainscore_file = open(score_dir+"/trainscore",'ab')

			trainscore_file.write(s)
			trainscore_file.close()
	
			# test on development data
			precision,recall,f_dev = self.evaluate(devdata, self.neg_label, self.ismention)
			s = "dev   score  ........  epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f_dev)+"\n"

			print s

			

			devscore_file = open(score_dir+"/devscore",'ab')

			devscore_file.write(s)
			devscore_file.close()

			
			# test on test data
			precision,recall,f = self.evaluate(testdata, self.neg_label,self.ismention)
			s = "test  score  .........  epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"

			print s
			testscore_file = open(score_dir+"/testscore",'ab')

			testscore_file.write(s)
			testscore_file.close()

			
			if f_dev > bestf and epoch > -1:
				bestmodel = score_dir+"/modelfile_"+str(epoch)
				bestf = f_dev

			
			


			
		self.loadmodel(bestmodel)
		return bestmodel


	def lang_eva(self, data):

		original = self.get_droppro()
		self.set_droppro(0.)

		right = 0.
		for ins in data:
			out = self.evafunc(ins[0])
			predict = np.argmax(out, axis = 1)
			gold = ins[1]
			if predict[len(predict)/2] == gold[len(gold)/2]:
				right += 1


		self.set_droppro(original)


		pre = right / len(data)
		return pre



	def evaluate(self, edata,neg_label,ismention = False):
		if self.model_type == "softmax" or self.model_type == "softmax_reg":
			return self.evaluate1(edata, neg_label)
		elif self.model_type == "maxneg":
			return self.evaluate2(edata, neg_label)
		elif self.model_type == "maxout":
			return self.evaluate3(edata, neg_label)
		else: raise Exception


	def evaluate1(self,edata,neg_label):

		overall = 0.
		extract = 0.
		correct = 0.
		original = self.get_droppro()
		self.set_droppro(0.)

		for ins in edata:
			out = self.evafunc(ins[0])
			if self.istransition:
				trans = self.w["trans"].get_value()
				result = self.decode(out, trans)
				predict = result[0][0]
			else:
				predict = np.argmax(out, axis = 1)
			gold = ins[1]
			if self.ismention:
				tool = {1:7,2:0,3:6,5:4}
				golden = self.extract(gold, tool)
				preden = self.extract(predict, tool)

				overall += 	len(golden)
				correct +=  self.common_num(golden, preden)
				extract += len(preden)

			else:
				overall += 	sum(1 for item in gold if item != neg_label)
				correct += sum(1 for index, item in enumerate(predict) if item != neg_label and item == gold[index])
				extract += sum(1 for item in predict if item != neg_label)

		print "extract  ..... "+str(extract)+"  correct  ......... "+str(correct)+ "  overall  .........  "+str(overall)
		precision = correct / (extract+1e-7)
		recall = correct /  overall
		f = 2./(1./(precision+1e-7)+1./(recall+1e-7))

		self.set_droppro(original)

		return precision, recall, f
	def extract(self, sen, tool):
		entities = []
		i = 0
		while i < len(sen):
			if sen[i] in tool:
				mid = tool[sen[i]]
				k = i+1
				while k < len(sen) and sen[k] == mid : k = k+1
				entities.append((i, k, mid))
				i = k - 1

			i += 1

		return entities

	def common_num(self, ens1,ens2):
		common = 0 
		for e1 in ens1:
			for e2 in ens2:
				if e1[0] == e2[0] and e1[1] == e2[1] and e1[2] == e2[2]:
					common +=1
					break
		return common





	def evaluate2(self,edata,neg_label):

		overall = 0.
		extract = 0.
		correct = 0.
		original = self.get_droppro()
		self.set_droppro(0.)

		for ins in edata:
			pos, neg = self.evafunc(ins[0])
			out = np.concatenate((pos,neg), axis = 1)
			predict = np.argmax(out, axis = 1)
			predict = [item if item < neg_label else neg_label for item in predict]

			gold = ins[1]

			overall += 	sum(1 for item in gold if item != neg_label)
			correct += sum(1 for index, item in enumerate(predict) if item != neg_label and item == gold[index])
			extract += sum(1 for item in predict if item != neg_label)

		precision = correct / (extract+1e-7)
		recall = correct /  overall
		f = 2./(1./(precision+1e-7)+1./(recall+1e-7))

		self.set_droppro(original)

		return precision, recall, f

	def evaluate3(self,edata,neg_label,isstat = True):

		overall = 0.
		extract = 0.
		correct = 0.
		original = self.get_droppro()
		self.set_droppro(0.)
		negtypes = {}
		for ins in edata:
			pos, neg = self.evafunc(ins[0])
			if isstat:
				maxpos = np.argmax(pos, axis = 1)
				maxneg = np.argmax(neg, axis = 1)
				for index in xrange(len(maxpos)):
					pos_index = maxpos[index]
					neg_index = maxneg[index]
					if neg[index][neg_index] > pos[index][pos_index]:
						negtypes.setdefault(neg_index, 0)
						negtypes[neg_index] += 1
			maxneg = np.max(neg,axis = 1)
			maxneg = maxneg.reshape((len(maxneg), 1))

			out = np.concatenate((pos,maxneg), axis = 1)
			predict = np.argmax(out, axis = 1)
		
			gold = ins[1]

			overall += 	sum(1 for item in gold if item != neg_label)
			correct += sum(1 for index, item in enumerate(predict) if item != neg_label and item == gold[index])
			extract += sum(1 for item in predict if item != neg_label)

		precision = correct / (extract+1e-7)
		recall = correct /  overall
		f = 2./(1./(precision+1e-7)+1./(recall+1e-7))

		self.set_droppro(original)
		print negtypes

		return precision, recall, f



	def score_labels(self, data, isstatistic = False):
		# 返回每一个实例类别的概率分布

		#original = self.get_droppro()
		#self.set_droppro(0.)
		
		result = []

		negtypes = {}
		for ins in data:
			if self.model_type == "softmax" or self.model_type == "softmax_reg":
				pros = self.evafunc(ins[0])
				result.append(pros)
			elif self.model_type == "maxneg":
				pos, neg = self.evafunc(ins[0])
				result.append(np.concatenate((pos,neg), axis = 1))
			elif self.model_type == "maxout":
				pos, neg = self.evafunc(ins[0])
				if isstatistic:
					maxpos = np.argmax(pos, axis = 1)
					maxneg = np.argmax(neg, axis = 1)
					for index in xrange(len(maxpos)):
						pos_index = maxpos[index]
						neg_index = maxneg[index]

						if neg[index][neg_index] > pos[index][pos_index]:
							negtypes.setdefault(neg_index, 0)
							negtypes[neg_index] += 1

				scores = np.concatenate((pos,np.max(neg, axis = 1).reshape((len(ins[0]),1))), axis = 1)
				scores = np.exp(scores)
				mysum = np.sum(scores, axis = 1)
				mysum = mysum.reshape((len(mysum),1))
				pro = scores / mysum
				result.append(pro)
			else: raise Exception

		#self.set_droppro(original)
		print negtypes
		print "result  num  :  "+str(len(result))
		return result

	

	def create_batch1(self, rawdata, active_data, seq_num, padding_id, token_num = 10):

		selectdata = []

		for ins in rawdata:
			token_randstart = random.randint(0,len(ins[1]) - token_num)
			y = [-1]*token_randstart + ins[1][token_randstart:token_randstart+token_num] + [-1]*(len(ins[1]) - token_randstart - token_num)
			selectdata.append([(ins[0], y)])

		newdata = selectdata

		for i in xrange(len(active_data) - 1):
			randindex = random.randint(i+1, len(active_data) - 1)
			active_data[i], active_data[randindex] = active_data[randindex], active_data[i]

		if active_data:
			adata = [active_data[i:i + token_num] for i in xrange(0,len(active_data),token_num)]# 使得每一个list内的实例数等于 token_num 
			newdata += adata
			print "active data  groups number  .......... "+str(len(adata))
			print "active data group  size ........... "+str(len(adata[0]))


		for i in xrange(len(newdata) - 1):
			randindex = random.randint(i+1, len(newdata) - 1)
			newdata[i], newdata[randindex] = newdata[randindex], newdata[i]


		batches = []	
		for i in xrange(0,len(newdata), seq_num):

			instances = [] 
			for item in newdata[i:i+seq_num]:
				instances.extend(item)


			max_len = max(len(item[0]) for item in instances)
			xlist = []
			ylist = []
			for ins in instances:
				xlist.append(ins[0]+[padding_id]*(max_len - len(ins[0])))
				ylist.append(ins[1]+[-1]*(max_len - len(ins[1])))
				assert len(xlist[-1]) == len(ylist[-1])


			
			batches.append((np.asarray(xlist, dtype = np.int32), np.asarray(ylist, dtype = np.int32)))


		return batches


	def create_batch2(self, data, batch_size):
		self.randomdata(data)

		def split(group):
			xgroup = [item[0] for item in group]
			ygroup = [item[1] for item in group]
			return (xgroup, ygroup)
		batches = [split(data[i:i+batch_size]) for i in xrange(0,len(data), batch_size)]

		return batches








	def randomdata(self, data):
		for i in xrange(len(data) - 1):
			randindex = random.randint(i+1, len(data) - 1)
			data[i], data[randindex] = data[randindex], data[i]


	def logp_loss1(self,x,y, fake_label, ismid = True):
		#y 中 -1 为 不计算loss的实例的label， fake_label = -1 用来产生mask掩盖掉 label为 -1 的实例
		# 返回平均 -logp 损失
		y = y.dimshuffle((1,0))
		inx = x.dimshuffle((1,0))

		if ismid:
			scores  = self.structure1(inx, ispro = False) #soft max layer output
			scores = scores[self.mid_id]
			y = y[self.mid_id]
			pre_y = y[self.mid_id - 1]
			if self.istransition:
				scores += self.w["trans"][pre_y]

			pro =  T.nnet.softmax(scores)
			losslist = T.nnet.categorical_crossentropy(pro, y)
			return T.sum(losslist)/losslist.size

		
		else:
			mask = T.neq(y,fake_label)
			y = y + T.eq(y, fake_label)
		
			#seq * batch * class
			pro  = self.structure1(inx) #soft max layer output
			
			pro = pro.reshape((pro.shape[0]*pro.shape[1], pro.shape[2]))

			y = y.flatten()

			losslist = T.nnet.categorical_crossentropy(pro, y)

			losslist = losslist.reshape(mask.shape)
			losslist = losslist*mask

			return T.sum(losslist)/T.sum(mask)




	def logp_reg(self,x,y, fake_label, low_weight,high_weight, variance = 0 ,nc = 1):# plus contractive regularization
		y = y.dimshuffle((1,0))
		inx = x.dimshuffle((1,0))
	

		#seq * batch * class
		mappinglist , pro = self.structure_reg(inx, variance = variance, nc =nc) #soft max layer output
		em, hid1 = mappinglist[0]
		
		pro = pro[self.mid_id]
		hid1 = hid1[self.mid_id]
		em = em[self.mid_id]



		y = y[self.mid_id]

		losslist = T.nnet.categorical_crossentropy(pro, y)

		#jaccobi reg
		hid1 = hid1.flatten()
		grad_tensor = theano.gradient.jacobian(hid1, em, disconnected_inputs = 'ignore')



		Hessian_norm = 0
		for item in mappinglist[1:]:
			bias_em = item[0]
			bias_hid1 = item[1]
			bias_hid1 = bias_hid1[self.mid_id]
			bias_em = bias_em[self.mid_id]
			bias_hid1 = bias_hid1.flatten()
			bias_grad_tensor = theano.gradient.jacobian(bias_hid1, bias_em, disconnected_inputs = 'ignore')
			Hessian_norm += T.sum((grad_tensor - bias_grad_tensor)**2)

		Hessian_norm /= nc


	
		loss = (T.sum(losslist) + low_weight*T.sum(grad_tensor**2) + high_weight*Hessian_norm)/losslist.size

		return loss

		
	def structure_reg(self,x, ispro = True, variance = 1, nc = 1): # ||J(x)-J(x+e)|| + ||J(x)||
		#BLSTM + FC + softmax
		em = self.embedLayer(x, self.dic)
		hid1 = self.BLSTMLayer(em, self.w, "layer_1", (self.net_size[0], self.net_size[1]))

		bias_ems = []
		bias_hid1s = []
		for i in xrange(nc):
			bias_ems.append(em + self.srng.normal(em.shape, std  = variance))
			bias_hid1s.append(self.BLSTMLayer(bias_ems[-1], self.w, "layer_1", (self.net_size[0], self.net_size[1])))

		hids = self.applyDropout2list([hid1]+ bias_hid1s)

		hid1 = hids[0]
		bias_hid1s = hids[1:]


		if self.midstr == "LSTM_forward":
			hid2 = self.applyDropout(self.forwardLayer(hid1,self.w, "layer_2", self.nonlinear["tanh"], (self.net_size[1], self.net_size[2])))
		elif self.midstr == "LSTM_LSTM":
			hid2 = self.applyDropout(self.BLSTMLayer(hid1, self.w, "layer_2", (self.net_size[1], self.net_size[2])))
		else:
			raise Exception

		if ispro:
			out = self.softmaxLayer(hid2, self.w, "layer_3", (self.net_size[2], self.net_size[3]))
		else:
			def identity(x):
				return x
			out = self.forwardLayer(hid2, self.w, "layer_3",identity, (self.net_size[2], self.net_size[3]))

		return zip([em] + bias_ems, [hid1] + bias_hid1s), out



	def structure1(self,x, ispro = True,isall = False):
		#BLSTM + FC + softmax
		em = self.embedLayer(x, self.dic)
		hid1 = self.applyDropout(self.BLSTMLayer(em, self.w, "layer_1", (self.net_size[0], self.net_size[1])))
		if self.midstr == "LSTM_forward":
			hid2 = self.applyDropout(self.forwardLayer(hid1,self.w, "layer_2", self.nonlinear["tanh"], (self.net_size[1], self.net_size[2])))
		elif self.midstr == "LSTM_LSTM":
			hid2 = self.applyDropout(self.BLSTMLayer(hid1, self.w, "layer_2", (self.net_size[1], self.net_size[2])))
		else:
			raise Exception

		if ispro:
			out = self.softmaxLayer(hid2, self.w, "layer_3", (self.net_size[2], self.net_size[3]))
		else:
			def identity(x):
				return x
			out = self.forwardLayer(hid2, self.w, "layer_3",identity, (self.net_size[2], self.net_size[3]))

		if isall:
			return [em, hid1, hid2 , out]
		else:
			return out 


	def logp_loss2(self,x,y, fake_label, neg_label, ismax = True):# neg_label is the maximum label id
		y = y.dimshuffle((1,0))
		inx = x.dimshuffle((1,0))
		fake_mask = T.neq(y, fake_label)
		y = y*fake_mask


		pos_mask = T.and_(fake_mask, T.le(y, neg_label-1))
		neg_mask = T.ge(y, neg_label)
		iny = y*pos_mask

		pos_pro, neg_pro = self.structure2(inx)

		if ismax:
			neg_pro = T.max(neg_pro, axis = -1)
			#pos_pro : sequence * batch * pos label
			#neg_pro :sequence *batch
			pos_logp = T.nnet.categorical_crossentropy(pos_pro.reshape((pos_pro.shape[0]*pos_pro.shape[1], pos_pro.shape[2])), iny.flatten())
			# sequence * batch 
			pos_logp = pos_logp.reshape(y.shape)*pos_mask  
			pos_loss = T.sum(pos_logp)
			neg_loss = 0 -  T.sum(T.log(neg_pro)*neg_mask)
			loss = (pos_loss + neg_loss)/ (T.sum(pos_mask)+T.sum(neg_mask))

		else:
			pro = T.concatenate((pos_pro, neg_pro), axis = 2)
			pro = pro.reshape((pro.shape[0]*pro.shape[1], pro.shape[2]))

			y = y.flatten()
			losslist = T.nnet.categorical_crossentropy(pro, y)
			losslist = losslist.reshape(fake_mask.shape)
			losslist = losslist*fake_mask

			loss = T.sum(losslist) / T.sum(fake_mask)



		return loss


	def logp_loss3(self, x, y, fake_label,neg_label, pos_ratio = 0.5): #adopt maxout  for  negative   
		# pos_rati0  means  pos examples weight (0.5 means  equal 1:1)


		print "adopt  positives  weight  ............. "+str(pos_ratio)
		y = y.dimshuffle((1,0))
		inx = x.dimshuffle((1,0))
		fake_mask = T.neq(y, fake_label)
		y = y*fake_mask

		pos_mask = T.and_(fake_mask, T.le(y, neg_label-1))*pos_ratio
		neg_mask = T.ge(y, neg_label)*(1- pos_ratio)


		pos_score, neg_score = self.structure2(inx,False)
		maxneg = T.max(neg_score, axis = -1)

		scores = T.concatenate((pos_score, maxneg.dimshuffle((0,1,'x'))), axis = 2)

		d3shape = scores.shape

		#seq*batch , label
		scores = scores.reshape((d3shape[0]*d3shape[1],  d3shape[2]))
		pro = T.nnet.softmax(scores)

		_logp = T.nnet.categorical_crossentropy(pro, y.flatten())

		_logp = _logp.reshape(fake_mask.shape)

		loss = (T.sum(_logp*pos_mask)+ T.sum(_logp*neg_mask))/ (T.sum(pos_mask)+T.sum(neg_mask))
		pos_loss = T.sum(_logp*pos_mask)
		neg_loss = T.sum(_logp*neg_mask)


		return loss, pos_loss, neg_loss




	def structure2(self,x,ispro = True):#BLSTM+ FC + softmax(pos) && maxout(neg)
		em = self.embedLayer(x, self.dic)

		hid1 = self.applyDropout(self.BLSTMLayer(em, self.w, "layer_1", (self.net_size[0], self.net_size[1])))

		hid2 = self.applyDropout(self.forwardLayer(hid1,self.w, "layer_2", self.nonlinear["tanh"], (self.net_size[1], self.net_size[2])))
		sw = self.set_para(self.w,"softw",shared32(1./np.sqrt(self.net_size[2])*np.random.randn(self.net_size[2],self.net_size[3][0])))
		sb = self.set_para(self.w,"softb",shared32(np.random.randn(self.net_size[3][0])))


		mw = self.set_para(self.w,"maxw",shared32(1./np.sqrt(self.net_size[2])*np.random.randn(self.net_size[2],self.net_size[3][1])))


		mb = self.set_para(self.w,"maxb",shared32(np.random.randn(self.net_size[3][1])))
		#mb = self.set_para(self.w,"maxb",shared32(0*np.random.randn(self.net_size[3][1])))

		if ispro:
			pos_score = T.exp(T.dot(hid2, sw)+sb)

			neg_score = T.exp(T.dot(hid2, mw)+mb)
			sumunit = T.sum(pos_score,axis = -1) + T.sum(neg_score, axis = -1)

			if x.ndim == 2:
				sumunit = sumunit.dimshuffle((0,1,'x'))
			elif x.ndim == 1:
				sumunit = sumunit.dimshuffle((0,'x'))
			pos_out = pos_score / sumunit
			neg_out =  neg_score / sumunit

		else:
			pos_out = T.dot(hid2, sw)+sb
			neg_out = T.dot(hid2, mw)+mb
			#neg_out = T.dot(hid2, mw)

		return pos_out, neg_out

		#pos_pro  : sequence * batch num * pos label num
		#neg_pro  : sequence *batch num *neg label num






	def set_para(self, w, key, random_init):
		if key in w:
			return w[key]

		else:
			print "random  weight .............. "+str(key)
			w[key] = random_init
			return random_init

 

 	def dumpmodel(self, fname,isemb = False):
 		model = {}
 		model["net_size"] = self.net_size
 		if isemb:
 			model["embedding"] = self.embeddic
 		model["model_weight"] = self.w
 		model["neg_label"] = self.neg_label
 		model["model_type"] = self.model_type
 		model["dropout"] = self.dropout_prob.get_value()
 		pickle.dump(model, open(fname, 'w+'))

 	def loadmodel(self, fname):
 		model = pickle.load(open(fname))
 		print "loading  best model .........."+str(fname)
 		self.w = model["model_weight"]

 	def initial_weight(self):
 		print "initiate model weight............."
 		for key in self.w:
 			value = self.w[key].get_value()
 			if value.ndim == 2:
 				self.w[key].set_value((1./np.sqrt(value.shape[0])*np.random.randn(*value.shape)).astype('float32'))
 			elif value.ndim == 1:
 				self.w[key].set_value(np.random.randn(*value.shape).astype('float32'))
 			else:
 				raise Exception

	def regmaxw(self,l2):
		print "adopt  maxw wdecay ............... "+str(l2)
		return T.sum(self.w["maxw"]**2)*l2
 		






  		 

