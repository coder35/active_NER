# coding=utf-8
import theano.tensor as T
import theano
import numpy as np
from BLSTM import BLSTMmodel
import pickle
import json
import math
def shared32(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)


class semimodel(BLSTMmodel):
    #wordnum is the number of output word type
    #padding_id is the id of output sentence START ,END and padding of output sentece
    def __init__(self,tag_num,emb_num, net_size,semiweight = 0, typenum = 0,model_type = "supervised",wordnum = 0, LMweight = 0,hsoftmax = False,padding_id =0,  isembed = False,layerid=1,
                 hinge  = 1.0, dropout = 0., mweight = 0.9, lrate = 0.01, opt = "momentum", wdecay = 0., fix_emb = False, embeddic = None, premodel = None):
        BLSTMmodel.__init__(self, tag_num,emb_num, net_size, dropout, mweight, lrate, opt, wdecay,fix_emb, embeddic, premodel)
        self.superw = {}
        self.unsuperw = {}
        self.LMweight = LMweight
        self.hsoftmax = hsoftmax
        self.model_type = model_type
        self.padding_id = padding_id
        for key in self.w:
            self.superw[key] = self.w[key]
        print "init semisupervised  weight ...........................?"
        if model_type == "mullabel":
            print "model type is  indeed  ............. multiple label"
            if "mulw" not in self.w:
                self.w["mulw"] = self.unsuperw["mulw"] = shared32(1./np.sqrt(net_size[layerid])*np.random.randn(net_size[layerid],typenum))
                self.w["mulb"] = self.unsuperw["mulb"] = shared32(np.random.randn(typenum))
                print "init multiple label layer weight ..........................."
            
            self.passweights(self.superw, self.unsuperw, layerid)#将embedding loss  需要用到的shared variables 传递过去

        elif model_type == "language":
            print "model type is indeed  ..........language"
            if self.hsoftmax:

                self.hshape = (int(math.sqrt(wordnum)), wordnum / int(math.sqrt(wordnum)) + 1)
                assert self.hshape[0] * self.hshape[1] >= wordnum
                self.w["posLMw1"] = self.unsuperw["posLMw1"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(net_size[layerid]/2, self.hshape[0]))
                self.w["posLMw2"] = self.unsuperw["posLMw2"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(self.hshape[0],net_size[layerid]/2, self.hshape[1]))
                self.w["posLMb1"] = self.unsuperw["posLMb1"] = shared32(np.random.randn(self.hshape[0]))
                self.w["posLMb2"] = self.unsuperw["posLMb2"] = shared32(np.random.randn(self.hshape[0], self.hshape[1]))

                self.w["negLMw1"] = self.unsuperw["negLMw1"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(net_size[layerid]/2, self.hshape[0]))
                self.w["negLMw2"] = self.unsuperw["negLMw2"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(self.hshape[0],net_size[layerid]/2, self.hshape[1]))
                self.w["negLMb1"] = self.unsuperw["negLMb1"] = shared32(np.random.randn(self.hshape[0]))
                self.w["negLMb2"] = self.unsuperw["negLMb2"] = shared32(np.random.randn(self.hshape[0], self.hshape[1]))
            else:

                self.w["posLMw"] = self.unsuperw["posLMw"] = shared32(1./np.sqrt(net_size[layerid]/2)* np.random.randn(net_size[layerid]/2, wordnum))
                self.w["posLMb"] = self.unsuperw["posLMb"] = shared32(np.random.randn(wordnum))
                self.w["negLMw"] = self.unsuperw["negLMw"] = shared32(1./np.sqrt(net_size[layerid]/2)* np.random.randn(net_size[layerid]/2, wordnum))
                self.w["negLMb"] = self.unsuperw["negLMb"] = shared32(np.random.randn(wordnum))
            self.passweights(self.superw, self.unsuperw, layerid)#将embedding loss  需要用到的shared variables 传递过去

        elif model_type == "multask":
            print "model type is indeed ..............mult-task "
            self.w["mulw"] = self.unsuperw["mulw"] = shared32(1./np.sqrt(net_size[layerid])*np.random.randn(net_size[layerid],typenum))
            self.w["mulb"] = self.unsuperw["mulb"] = shared32(np.random.randn(typenum))
            if self.hsoftmax:

                self.hshape = (int(math.sqrt(wordnum)), wordnum / int(math.sqrt(wordnum)) + 1)
                assert self.hshape[0] * self.hshape[1] >= wordnum
                self.w["posLMw1"] = self.unsuperw["posLMw1"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(net_size[layerid]/2, self.hshape[0]))
                self.w["posLMw2"] = self.unsuperw["posLMw2"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(self.hshape[0],net_size[layerid]/2, self.hshape[1]))
                self.w["posLMb1"] = self.unsuperw["posLMb1"] = shared32(np.random.randn(self.hshape[0]))
                self.w["posLMb2"] = self.unsuperw["posLMb2"] = shared32(np.random.randn(self.hshape[0], self.hshape[1]))

                self.w["negLMw1"] = self.unsuperw["negLMw1"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(net_size[layerid]/2, self.hshape[0]))
                self.w["negLMw2"] = self.unsuperw["negLMw2"] = shared32(1./np.sqrt(net_size[layerid]/2)*np.random.randn(self.hshape[0],net_size[layerid]/2, self.hshape[1]))
                self.w["negLMb1"] = self.unsuperw["negLMb1"] = shared32(np.random.randn(self.hshape[0]))
                self.w["negLMb2"] = self.unsuperw["negLMb2"] = shared32(np.random.randn(self.hshape[0], self.hshape[1]))
            else:

                self.w["posLMw"] = self.unsuperw["posLMw"] = shared32(1./np.sqrt(net_size[layerid]/2)* np.random.randn(net_size[layerid]/2, wordnum))
                self.w["posLMb"] = self.unsuperw["posLMb"] = shared32(np.random.randn(wordnum))
                self.w["negLMw"] = self.unsuperw["negLMw"] = shared32(1./np.sqrt(net_size[layerid]/2)* np.random.randn(net_size[layerid]/2, wordnum))
                self.w["negLMb"] = self.unsuperw["negLMb"] = shared32(np.random.randn(wordnum))

            self.passweights(self.superw, self.unsuperw, layerid)#将embedding loss  需要用到的shared variables 传递过去


        '''
        self.unsuperw = {}
        for layerid, laydim in zip(emlayerid,emlayerdim):
            self.unsuperw["elayer"+str(layerid)] = shared32(1./np.sqrt(net_size[layerid])*np.random.randn(net_size[layerid],laydim))
        for key in embeddic:
            self.unsuperw[key] = self.superw[key]
        '''

        self.wdecay = wdecay
        self.semiweight = semiweight #magnet loss weight
        self.layerid = layerid# compute mutiple class loss from layer "layerid"

    def train_ready(self):
        if self.model_type == "supervised":
            self.supervised = BLSTMmodel.train_ready(self)# 输入 带有标签的group执行更新并输出-logp 损失
        elif self.model_type == "mullabel":
            self.unsupervised =  self.mulclassfunc(self.layerid, self.wdecay)
            self.supervised = BLSTMmodel.train_ready(self)# 输入 带有标签的group执行更新并输出-logp 损失

        elif self.model_type == "multask":
            self.unsupervised =  self.multaskfunc(self.layerid,self.wdecay, self.LMweight)
            self.supervised = BLSTMmodel.train_ready(self)# 输入 带有标签的group执行更新并输出-logp 损失

        #self.semisupervised = self.jointfunc(self.layerid, self.semiweight, self.wdecay) #输入a list of group（clusters） 每个group不带标签但标明实体位置， 加上带有标签的supergroup


    def unsup_evaluate_ready(self):
        self.unsupeva = self.mulclasseva(self.layerid)


    def unsup_evaluate(self, group):

        finput = self.raw2input2(group)
        result = self.unsupeva(*finput[:-1])
        correct = 0
        for pre, gold in zip(result, finput[-1]):
            pre = [1 if item > 0.5 else -1 for item in pre]
            for pitem, gitem in zip(pre,gold):
                if pitem == gitem:
                    correct += 1
        return correct
    def superUpdate(self,group):# list of tuples of x and y
        finput = self.raw2input1(group) # different models has separate data processing function
        result = self.supervised(*finput)
        return result
    def unsuperUpdate(self,group):
        if self.model_type == "multask":
            finput = self.raw2input3(group)
        else:
            finput = self.raw2input2(group)

        result = self.unsupervised(*finput)
        return result

    def semiUpdate(self, unsugroup, supergroup):
        iin = self.raw2input2(unsugroup)
        iin.extend(self.raw2input1(supergroup))

        result = self.semisupervised(*iin)
        return result

    def computeEmb(self,group):
        iin  = self.raw2input2(group)
        embeddings = self.embedmap(*iin)
        return embeddings
    '''
    def embedfunc(self, layerid):
        x = []
        for i in range(self.emb_num):
            x.append(T.tensor3(dtype = 'int32'))
        y = T.tensor3(dtype = 'int8')
        embedding = self.embed(x,y,layerid)

        print "embedding type  : "+str(embedding.type)
        embedmap = theano.function(x+[y], embedding)
        return embedmap

    '''
    def embed(self,x, y, kth):

        hidden = self.hidden_k(x,self.superw,self.dicw, kth)
        size = y.ndim

        y = T.addbroadcast(y,size - 1)
        embedding = T.sum(hidden*y,0)/T.addbroadcast(T.cast(T.sum(y,0), 'int16'), size - 2)
        return embedding



    def mulclasseva(self,layerid):

        x = []
        for j in range(self.emb_num):
            x.append(T.tensor3(dtype = 'int32'))
        y = T.tensor3(dtype = 'int8')

        iin = []
        iin.extend(x)
        iin.append(y)


        hidden = self.hidden_k(x,self.w,self.dicw,layerid )

        size = y.ndim
        y = T.addbroadcast(y,size - 1)
        #embedding = T.sum(hidden*y,0)/T.addbroadcast(T.cast(T.sum(y,0), 'int16'), size - 2)
        embedding = T.sum(hidden*y,0)/ T.addbroadcast(T.sum(y,0), size-2)

        pro = 1. / (1. + T.exp(0. - (T.dot(embedding, self.w["mulw"])+self.w["mulb"])))
        mulclasspro = theano.function(iin, pro)

        return mulclasspro

    def multaskfunc(self,layerid,wdecay, LMweight):
        #language model + multiple label classification
        # multi-label loss
        x = []
        for j in range(self.emb_num):
            x.append(T.tensor3(dtype = 'int32'))
        y = T.tensor3(dtype = 'int8')
        label = T.matrix(dtype = 'int8')
        nextwords = T.imatrix()

        iin = []
        iin.extend(x)
        iin.append(y)
        iin.append(label)
        iin.append(nextwords)

        mulloss, posLMloss, negLMloss = self.LMmulcloss(layerid,x,y,label,nextwords)

        loss = mulloss + LMweight*(posLMloss+negLMloss)+self.l2reg(self.unsuperw, wdecay)
        w = self.unsuperw
        witems = w.values()
        if not self.fix_emb:
            witems += self.dicw.values()

        g = T.grad(loss, witems)
        up = self.upda(g,witems,self.lrate, self.mweight,self.opt,self.fix_emb)

        mtaskfunc = theano.function(iin, loss, updates = up)
        return mtaskfunc

    def mulclassfunc(self, layerid, wdecay):
        # mult-label loss
        x = []
        for j in range(self.emb_num):
            x.append(T.tensor3(dtype = 'int32'))
        y = T.tensor3(dtype = 'int8')
        label = T.matrix(dtype = 'int8')

        iin = []
        iin.extend(x)
        iin.append(y)
        iin.append(label)


        wikiloss = self.mulclassloss(layerid,x,y,label)

        loss = wikiloss + self.l2reg(self.unsuperw, wdecay)

        w = self.unsuperw
        witems = w.values()
        if not self.fix_emb:
            witems += self.dicw.values()

        g = T.grad(loss, witems)
        up = self.upda(g,witems,self.lrate, self.mweight,self.opt,self.fix_emb)

        mulclassfunc = theano.function(iin, loss, updates = up)
        return mulclassfunc



    def jointfunc(self, layerid, semiweight, wdecay):
        #magnet loss and supervised loss

        # wiki loss
        x = []
        for j in range(self.emb_num):
            x.append(T.tensor3(dtype = 'int32'))
        y = T.tensor3(dtype = 'int8')
        label = T.matrix(dtype = 'int8')

        iin = []
        iin.extend(x)
        iin.append(y)
        iin.append(label)


        wikiloss = self.mulclassloss(layerid,x,y,label)

        #supervised log p

        groups_x  = []
        for i in range(self.emb_num):
            groups_x.append(T.tensor3(name = 'gx'+str(i), dtype = 'int32'))
        group_y = T.tensor3(name ='y', dtype = 'int8')
        iin.extend(groups_x)
        iin.append(group_y)

        logploss  = 0. - self.logp(groups_x,group_y,self.w,self.dicw)/groups_x[0].shape[1]


        loss = logploss + semiweight*wikiloss + self.l2reg(self.w, wdecay)

        witems = self.w.values()
        if not self.fix_emb:
            witems.extend(self.dicw.values())

        g = T.grad(loss, witems)
        up = self.upda(g,witems,self.lrate, self.mweight,self.opt,self.fix_emb)

        semifunc = theano.function(iin, loss, updates = up)

        return semifunc

    def difloss(self,kth,x,y,postags,negtags,hinge,typeembs,lastw = None,isembed = False):
        hidden = self.hidden_k(x,self.w,self.dicw, kth)

        size = y.ndim
        y = T.addbroadcast(y,size - 1)
        embedding = T.sum(hidden*y,0)/T.addbroadcast(T.cast(T.sum(y,0), 'int16'), size - 2)
        if isembed:
            #embedding = T.dot(T.cast(embedding,theano.config.floatX), lastw)
            embedding = T.dot(embedding, lastw)

        dis1 = typeembs[postags] - embedding
        numerator = T.sum(T.exp(0. - T.sum(dis1**2,dis1.ndim - 1)), 0)/postags.shape[0]

        dis2 = typeembs[negtags] - embedding
        denominator = T.sum(T.exp(0. - T.sum(dis2**2,dis2.ndim - 1)), 0)

        def marginloss(x): return x*(x>0)
        loss = T.sum(marginloss((0. - T.log(numerator/denominator) + hinge)))/numerator.shape[0]

        return loss

    def mulclassloss(self,kth,x,y,label):
        #mutiple label classification loss using wikidata for pretrain
        hidden = self.hidden_k(x,self.w,self.dicw,kth)
        print "hidden  type :  "+str(hidden.type)
        size = y.ndim
        y = T.addbroadcast(y,size - 1)
        embedding = T.sum(hidden*y,0)/T.addbroadcast(T.cast(T.sum(y,0), 'int16'), size - 2)
        #embedding = T.sum(hidden*y,0)/ T.addbroadcast(T.sum(y,0), size-2)
        print "embedding type  :  "+str(embedding.type)
        logloss = (0. - T.sum(T.log(1. / (1. + T.exp(0. - (T.dot(embedding, self.w["mulw"])+self.w["mulb"])*label)))))/embedding.shape[0]
        return logloss
    def LMmulcloss(self,kth,x,y,label,nextwords):
        #multiple label loss + language model loss
        #nextwords  = START + words + END
        hidden = self.hidden_k(x,self.w,self.dicw,kth)
        print "hidden  type :  "+str(hidden.type)
        size = y.ndim
        y = T.addbroadcast(y,size - 1)
        embedding = T.sum(hidden*y,0)/T.addbroadcast(T.cast(T.sum(y,0), 'int16'), size - 2)
        #embedding = T.sum(hidden*y,0)/ T.addbroadcast(T.sum(y,0), size-2)
        print "embedding type  :  "+str(embedding.type)
        mulloss = (0. - T.sum(T.log(1. / (1. + T.exp(0. - (T.dot(embedding, self.w["mulw"])+self.w["mulb"])*label)))))/embedding.shape[0]


        if self.hsoftmax:
            #pos language model
            hshape = self.hshape
            newhidden = hidden[:,:,:hidden.shape[2]/2].reshape((hidden.shape[0]*hidden.shape[1],hidden.shape[2]/2))

            smax_group = T.nnet.h_softmax(newhidden, newhidden.shape[0],self.wordnum, hshape[0], hshape[1], self.w["posLMw1"], self.b["posLMb1"], self.w["posLMw2"], self.w["posLMb2"], nextwords[2:].ravel())
            losslist = T.neg(T.log(smax_group.reshape(nextwords[2:].shape)))
            mask = T.cast(T.neq(nextwords[2:], self.padding_id), theano.config.floatX)
            losslist = losslist*mask
            posLMloss = T.cast(T.mean(T.sum(losslist,axis=0)), theano.config.floatX)

            #neg language model

            newhidden = hidden[:,:,hidden.shape[2]/2:].reshape((hidden.shape[0]*hidden.shape[1],hidden.shape[2]/2))

            smax_group = T.nnet.h_softmax(newhidden, newhidden.shape[0],self.wordnum, hshape[0], hshape[1], self.w["negLMw1"], self.b["negLMb1"], self.w["negLMw2"], self.w["negLMb2"], nextwords[:-2].ravel())
            losslist = T.neg(T.log(smax_group.reshape(nextwords[:-2].shape)))
            mask = T.cast(T.neq(nextwords[:-2], self.padding_id), theano.config.floatX)
            losslist = losslist*mask
            negLMloss = T.cast(T.mean(T.sum(losslist,axis=0)), theano.config.floatX)

        else:


            def categorical_loss(ihidden,words,w,b):
                scores = T.dot(ihidden,w)+b
                prep = T.exp(scores)/T.sum(T.exp(scores),1).dimshuffle(0,'x')
                loss = T.nnet.categorical_crossentropy(prep, words)
                return loss
            #newhidden = hidden.reshape((hidden.shape[0]*hidden.shape[1], hidden.shape[2]))
            #pos language model
            #prep = T.exp(T.dot(newhidden[:,:newhidden.shape[1]/2],self.w["posLMw"])+self.w["posLMb"])/T.sum(T.exp(T.dot(newhidden[:,:newhidden.shape[1]/2],self.w["posLMw"])+self.w["posLMb"]), 1).dimshuffle(0,'x')
            
            scores = T.dot(hidden[:,:,:hidden.shape[2]/2], self.w["posLMw"])+self.w["posLMb"]
            scores = scores.reshape((scores.shape[0]*scores.shape[1], scores.shape[2]))
            prep = T.exp(scores)/T.sum(T.exp(scores), scores.ndim - 1).dimshuffle((0,'x'))
            #(len*batch)
            losslist = T.nnet.categorical_crossentropy(prep,nextwords[2:].ravel())
            losslist = losslist.reshape(nextwords[2:].shape)
            
            #losslist, _ = theano.scan(fn = categorical_loss, sequences = [hidden[:,:,:hidden.shape[2]/2], nextwords[2:]], outputs_info = None, 
            #    non_sequences = [self.w["posLMw"], self.w["posLMb"]]) 



            mask = T.cast(T.neq(nextwords[2:], self.padding_id), theano.config.floatX)
            losslist = losslist*mask
            posLMloss = T.cast(T.mean(T.sum(losslist,axis=0)), theano.config.floatX)


            #neg language model
            #prep = T.exp(T.dot(newhidden[:,newhidden.shape[1]/2:],self.w["negLMw"])+self.w["negLMb"])/T.sum(T.exp(T.dot(newhidden[:,newhidden.shape[1]/2:],self.w["negLMw"])+self.w["negLMb"]), 1).dimshuffle(0,'x')
            
            scores = T.dot(hidden[:,:,hidden.shape[2]/2:], self.w["negLMw"])+self.w["negLMb"]
            scores = scores.reshape((scores.shape[0]*scores.shape[1], scores.shape[2]))
            prep = T.exp(scores)/T.sum(T.exp(scores), scores.ndim - 1).dimshuffle((0,'x'))
            #(len*batch)
            losslist = T.nnet.categorical_crossentropy(prep,nextwords[0:-2].ravel())
            losslist = losslist.reshape(nextwords[0:-2].shape)
            

            

            #losslist, _ = theano.scan(fn = categorical_loss, sequences = [hidden[:,:,hidden.shape[2]/2:], nextwords[:-2]], outputs_info = None, 
            #    non_sequences = [self.w["negLMw"], self.w["negLMb"]]) 

            mask = T.cast(T.neq(nextwords[0:-2], self.padding_id), theano.config.floatX)
            losslist = losslist*mask
            negLMloss = T.cast(T.mean(T.sum(losslist,axis=0)), theano.config.floatX)


        return mulloss, posLMloss, negLMloss
    def passweights(self, superw, unsuperw,k):

        for i in range(k):
            layerid = "layer_"+str(i+1)
            unsuperw["wxi"+layerid] = superw["wxi"+layerid]
            unsuperw["wxf"+layerid] = superw["wxf"+layerid]
            unsuperw["wxo"+layerid] = superw["wxo"+layerid]
            unsuperw["wx"+layerid] = superw["wx"+layerid]

            unsuperw["whi"+layerid] = superw["whi"+layerid]
            unsuperw["whf"+layerid] = superw["whf"+layerid]
            unsuperw["who"+layerid] = superw["who"+layerid]
            unsuperw["wh"+layerid] = superw["wh"+layerid]

            unsuperw["wci"+layerid] = superw["wci"+layerid]
            unsuperw["wcf"+layerid] = superw["wcf"+layerid]
            unsuperw["wco"+layerid] = superw["wco"+layerid]

            unsuperw["big"+layerid] = superw["big"+layerid]
            unsuperw["bfg"+layerid] = superw["bfg"+layerid]
            unsuperw["bog"+layerid] = superw["bog"+layerid]

            unsuperw["bx"+layerid] = superw["bx"+layerid]


            unsuperw["wxi_r"+layerid] = superw["wxi_r"+layerid]
            unsuperw["wxf_r"+layerid] = superw["wxf_r"+layerid]
            unsuperw["wxo_r"+layerid] = superw["wxo_r"+layerid]
            unsuperw["wx_r"+layerid] = superw["wx_r"+layerid]

            unsuperw["whi_r"+layerid] = superw["whi_r"+layerid]
            unsuperw["whf_r"+layerid] = superw["whf_r"+layerid]
            unsuperw["who_r"+layerid] = superw["who_r"+layerid]
            unsuperw["wh_r"+layerid] = superw["wh_r"+layerid]

            unsuperw["wci_r"+layerid] = superw["wci_r"+layerid]
            unsuperw["wcf_r"+layerid] = superw["wcf_r"+layerid]
            unsuperw["wco_r"+layerid] = superw["wco_r"+layerid]

            unsuperw["big_r"+layerid] = superw["big_r"+layerid]
            unsuperw["bfg_r"+layerid] = superw["bfg_r"+layerid]
            unsuperw["bog_r"+layerid] = superw["bog_r"+layerid]

            unsuperw["bx_r"+layerid] = superw["bx_r"+layerid]


    def printmodel(self,filename,isword):
        if isword:
            f_dic = open(filename+"_dic", 'wb')
            outdic = {}
            for key in self.dicw:
                outdic[key] = self.dicw[key].get_value()

            pickle.dump(outdic, f_dic)
            f_dic.close()

        f_w = open(filename+"_w", 'wb')
        outw = {}
        for key in self.w:
            outw[key] = self.w[key].get_value()

        pickle.dump(outw, f_w)
        f_w.close()

    '''
    def readmodel(self,filename,isword):

        if isword:
            f_dic = open(filename+"_dic",'rb')
            self.dicw = pickle.load(f_dic)

        f_w = open(filename+"_w", 'rb')
        self.w = pickle.load(f_w)
    '''
