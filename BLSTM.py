# coding=utf-8
import theano.tensor as T
import theano
import numpy as np
import pickle
from model import Model
import random
def shared32(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)





class BLSTMmodel(Model):

    def __init__(self, tag_num,emb_num, net_size,drop_pro, mweight = 0.9, lrate = 0.01, opt = "momentum", wdecay = 0., fix_emb = False, embeddic = None, premodel =None):
        Model.__init__(self,emb_num, wdecay,opt, mweight, lrate)
        def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
        def f_rectlin(x): return x*(x>0)
        def f_rectlin2(x): return x*(x>0) + 0.01 * x
        self.nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2}
        self.tag_num = tag_num
        self.net_size = net_size

        self.fix_emb = fix_emb
        print "whether  fixing  word embedding  :  "+str(self.fix_emb)

        self.w,self.dicw = self.init_w(embeddic, premodel)

        self.dropout_prob = shared32(drop_pro)
        self.srng = T.shared_randomstreams.RandomStreams(random.randint(0,9999))

    def train_ready(self):

        groups_x  = []
        for i in range(self.emb_num):
            groups_x.append(T.tensor3(name = 'gx'+str(i), dtype = 'int32'))
        group_y = T.tensor3(name ='y', dtype = 'int8')
        '''
        #debug
        theano.config.compute_test_value = 'warn'
        finput = self.raw2input(group) # different models has separate data processing function
        for i in range(len(groups_x)):
            groups_x[i].tag.test_value = finput[i]

        group_y.tag.test_value = finput[-1]
        #debug
        '''
        output = self.l2reg(self.w, self.wdecay) - self.logp(groups_x,group_y,self.w,self.dicw)/groups_x[0].shape[1]

        witems = self.w.values()
        if not self.fix_emb:
            witems += self.dicw.values()

        g = T.grad(output, witems)

        #for witem, gitem in zip(witems, g):
        #    print "weight type  :  "+str(witem.type)+ "  gradient type  :  "+str(gitem.type)

        mweight = self.mweight
        lrate = self.lrate
        up = self.upda(g,witems,lrate, mweight,self.opt,self.fix_emb)

        self.updatefunc = theano.function(groups_x+[group_y], output, updates = up)
        return self.updatefunc
        #self.debug_func = theano.function(groups_x, [self.debug_em,self.debug_result1,self.debug_final])

    def evaluate_ready(self):
        x  = []
        for i in range(self.emb_num):
            x.append(T.matrix(name = 'x'+str(i), dtype = 'int32'))
        self.cs_func  = theano.function(x, self.cscore(x,self.w,self.dicw))
        return self.cs_func
    def upAndEva(self,group):# list of tuples of x and y
        finput = self.raw2input1(group) # different models has separate data processing function
        #debug
        '''
        r = self.debug_func(finput[0])
        print "embedding shape :"+str(r[0].shape)
        print "..............................."
        print "one pass rnn result shape "+str(r[1].shape)
        print "..............................."
        print "BLSTM final result shape "+str(r[2].shape)
        '''

        result = self.updatefunc(*finput)
        return result


    def set_dropout(self,value):
        self.dropout_prob.set_value(value)

    def l2reg(self,w,wdecay): # only deal with laywise weights
        reg = 0
        for key in w:
            reg += T.sum(w[key]**2)

        return reg*wdecay


    def raw2input1(self,group):
        #堆积group中每个实例，将y的索引表示转化为one-hot 向量表示，产生theano函数符合条件的输入
        groups_x = []
        group_y = []

        selen = len(group[0][1]) #sequence length
        embnum = len(group[0][0])#number of differnet index group in x

        for i in xrange(embnum):
            groups_x.append([])
            for j in range(selen):
                groups_x[len(groups_x)-1].append([])
        for i in xrange(selen + 1):# the length of y is equal to that of x plus 1, the first one in y is used as start symbol
            group_y.append([])

        #print "sequence length   :   "+str(selen)
        for instance in group:# for each instance  eg. instance[0]:([[1,2],[2,3]],[[1],[2]])  instance[1] : [3 ,-1] 小于0代表padding的标签
            for i in xrange(len(instance[0])):
                indexgroup = instance[0][i]
                for j in xrange(selen):
                    groups_x[i][j].append(indexgroup[j])



            y = np.arange(self.tag_num*(selen + 1), dtype = np.int8).reshape((selen + 1,self.tag_num))*0

            for i in xrange(1, selen + 1):
                if instance[1][i-1] < 0:
                    for j in range(len(y[i])):
                        y[i][j] = 1

                else:
                    y[i][instance[1][i-1]] = 1

            for i in xrange(selen + 1):
                group_y[i].append(y[i])


        funcinput = [np.asarray(x, dtype = np.int32) for x in groups_x] + [np.asarray(group_y, dtype = np.int8)]

        return funcinput

    def raw2input2(self,group):
        #堆积group中每个实例，将y的索引表示转化为0,1 向量表示(1表示是实体，0表示不是实体），产生theano函数符合条件的输入
        groups_x = []
        group_y = []
        group_label = []

        selen = len(group[0][1]) #sequence length
        embnum = len(group[0][0])#number of differnet index group in x
        for i in range(embnum):
            groups_x.append([])
            for j in range(selen):
                groups_x[len(groups_x)-1].append([])
        for i in range(selen):
            group_y.append([])

        #print "sequence length   :   "+str(selen)
        for instance in group:# for each instance  eg. instance[0]:([[1,2],[2,3]],[[1],[2]])  instance[1] : [1 ,0] 1表示实体， 0代表padding的标签或者非实体
            for i in range(len(instance[0])):
                indexgroup = instance[0][i]
                assert len(indexgroup) == selen
                for j in range(selen):
                    groups_x[i][j].append(indexgroup[j])


            y = instance[1]
            for i in range(selen):
                group_y[i].append([y[i]])

            group_label.append(instance[2])

        groupinput = [np.asarray(x, dtype = np.int32) for x in groups_x] + [np.asarray(group_y, dtype = np.int8), np.asarray(group_label,dtype = np.int8)]
        return groupinput


    def raw2input3(self,group):

        groups_x = []
        group_y = []
        group_label = []
        group_genwords = []

        selen = len(group[0][1]) #sequence length
        embnum = len(group[0][0])#number of differnet index group in x
        outlen = len(group[0][3])

        assert selen + 2 == outlen
        for i in xrange(embnum):
            groups_x.append([])
            for j in xrange(selen):
                groups_x[len(groups_x)-1].append([])
        for i in xrange(selen):
            group_y.append([])

        for i in xrange(outlen):
            group_genwords.append([])

        #print "sequence length   :   "+str(selen)
        for instance in group:# for each instance  eg. instance[0]:([[1,2],[2,3]],[[1],[2]])  instance[1] : [1 ,0] 1表示实体， 0代表padding的标签或者非实体
            for i in xrange(len(instance[0])):
                indexgroup = instance[0][i]
                assert len(indexgroup) == selen
                for j in xrange(selen):
                    groups_x[i][j].append(indexgroup[j])


            y = instance[1]
            for i in xrange(selen):
                group_y[i].append([y[i]])

            group_label.append(instance[2])

            assert outlen == len(instance[3])
            for i in xrange(outlen):
                group_genwords[i].append(instance[3][i])


        groupinput = [np.asarray(x, dtype = np.int32) for x in groups_x] + [np.asarray(group_y, dtype = np.int8), np.asarray(group_label,dtype = np.int8), np.asarray(group_genwords, dtype = np.int32)]
        return groupinput




    def init_w(self, embdic = None, premodel = None):
        w = {}
        dicw = {}

        size = self.net_size
        n_t = self.tag_num
        if premodel == None:
            for i in range(len(size) - 2):
                layerid = "layer_"+str(i+1)
                w["wxi"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))
                w["wxf"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))
                w["wxo"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))
                w["wx"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))

                w["whi"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))
                w["whf"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))
                w["who"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))
                w["wh"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))
                w["wci"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["wcf"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["wco"+layerid] = shared32(np.random.randn(size[i+1]/2))

                w["big"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["bfg"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["bog"+layerid] = shared32(np.random.randn(size[i+1]/2))

                w["bx"+layerid] = shared32(np.random.randn(size[i+1]/2))


                w["wxi_r"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))
                w["wxf_r"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))
                w["wxo_r"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))
                w["wx_r"+layerid] = shared32(1./np.sqrt(size[i])*np.random.randn(size[i],size[i+1]/2))

                w["whi_r"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))
                w["whf_r"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))
                w["who_r"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))
                w["wh_r"+layerid] = shared32(1./np.sqrt(size[i+1]/2)*np.random.randn(size[i+1]/2,size[i+1]/2))


                w["wci_r"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["wcf_r"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["wco_r"+layerid] = shared32(np.random.randn(size[i+1]/2))



                w["big_r"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["bfg_r"+layerid] = shared32(np.random.randn(size[i+1]/2))
                w["bog_r"+layerid] = shared32(np.random.randn(size[i+1]/2))

                w["bx_r"+layerid] = shared32(np.random.randn(size[i+1]/2))

            w["A"] = shared32(np.random.randn(n_t,n_t))

            w["wo"] = shared32(1./np.sqrt(size[-2])*np.random.randn(size[-2], size[len(size)-1]))
            w["bo"] = shared32(np.random.randn(size[len(size)-1]))

            if embdic == None:
                raise Exception("embdedding dic not initialization")
            else:
                for key in embdic:
                    dicw[key] = theano.shared(np.asarray(embdic[key],dtype = 'float32'))


        else :
            #problem code

            for key in premodel:
                #if key.startswith("w")  or  key.startswith("b"):
                w[key] = shared32(premodel[key])

            #if "A" not in w:

                #w["A"] = shared32(np.random.randn(n_t,n_t))
                #w["wo"] = shared32(1./np.sqrt(size[-2])*np.random.randn(size[-2], size[-1]))
                #w["bo"] = shared32(np.random.randn(size[-1]))

            if embdic == None:
                raise Exception("embdedding dic not initialization")
            else:
                for key in embdic:
                    dicw[key] = theano.shared(np.asarray(embdic[key],dtype = 'float32'))

        return w,dicw


    def applyDropout(self,x):

        d = 1-self.dropout_prob
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = x.shape
            )
        mask = T.cast(mask, theano.config.floatX) / d
        return x*mask


    def embedLayer(self,x,dic):
        return dic[x].flatten(x.ndim)



    def BLSTMLayer(self,x,w,layerid):
        wxi = w["wxi"+layerid]
        whi = w["whi"+layerid]
        wci = w["wci"+layerid]
        wxf = w["wxf"+layerid]
        whf = w["whf"+layerid]
        wcf = w["wcf"+layerid]
        wx = w["wx"+layerid]
        wh = w["wh"+layerid]#输入
        wxo = w["wxo"+layerid]
        who = w["who"+layerid]
        wco = w["wco"+layerid]

        big = w["big"+layerid]
        bfg = w["bfg"+layerid]
        bog = w["bog"+layerid]
        bx = w["bx"+layerid]##输入
#反向权重
        wxi_r = w["wxi_r"+layerid]
        whi_r = w["whi_r"+layerid]
        wci_r = w["wci_r"+layerid]
        wxf_r = w["wxf_r"+layerid]
        whf_r = w["whf_r"+layerid]
        wcf_r = w["wcf_r"+layerid]
        wx_r = w["wx_r"+layerid]
        wh_r = w["wh_r"+layerid]#输入
        wxo_r = w["wxo_r"+layerid]
        who_r = w["who_r"+layerid]
        wco_r = w["wco_r"+layerid]

        big_r = w["big_r"+layerid]
        bfg_r = w["bfg_r"+layerid]
        bog_r = w["bog_r"+layerid]
        bx_r = w["bx_r"+layerid]##输入

        sig= self.nonlinear["sigmoid"]
        tan =self.nonlinear["tanh"]
        def forward_pass(lx, h, c, wxi, whi, wci, wxf, whf, wcf, wx, wh,  wxo, who, wco, big, bfg, bog,bx):
            igate = T.dot(lx,wxi)+T.dot(h,whi)+c*wci+big
            fgate = T.dot(lx,wxf)+T.dot(h,whf)+c*wcf+bfg
            inpu_t = T.dot(lx,wx)+T.dot(h,wh)+bx
            new_c = sig(fgate)*c + sig(igate)*tan(inpu_t)
            ogate = T.dot(lx,wxo)+T.dot(h,who)+new_c*wco+bog
            new_h = sig(ogate)*tan(new_c)
            return [new_h, new_c ]

        def backward_pass(lx, for_h, h, c, wxi, whi, wci, wxf, whf, wcf, wx, wh,  wxo, who, wco, big, bfg, bog,bx):
            igate = T.dot(lx,wxi)+T.dot(h,whi)+c*wci+big
            fgate = T.dot(lx,wxf)+T.dot(h,whf)+c*wcf+bfg
            inpu_t = T.dot(lx,wx)+T.dot(h,wh)+bx
            new_c = sig(fgate)*c + sig(igate)*tan(inpu_t)
            ogate = T.dot(lx,wxo)+T.dot(h,who)+new_c*wco+bog
            new_h = sig(ogate)*tan(new_c)
            all_h = T.concatenate([for_h,new_h], axis = for_h.ndim - 1)

            return [new_h, new_c, all_h]


        initial = T.zeros_like(T.dot(x[0], wxi))
        result,up = theano.scan(fn = forward_pass, sequences = x, outputs_info = [initial,initial],
                                non_sequences = [wxi, whi, wci,wxf, whf, wcf, wx, wh,  wxo, who, wco, big, bfg, bog,bx])

        self.debug_result1 = result[0]
        result_r,up_r = theano.scan(fn = backward_pass, sequences = [x, result[0]],
                                    outputs_info = [initial, initial, None],
                                non_sequences = [wxi_r, whi_r, wci_r, wxf_r, whf_r, wcf_r, wx_r, wh_r,  wxo_r, who_r, wco_r,big_r, bfg_r, bog_r,bx_r], go_backwards  = True)


        final, _ = theano.scan(lambda x:x, sequences = result_r[2], outputs_info = None, go_backwards = True)
        self.debug_final = final
        return final




    def hidden_k(self,x,w,dicw,k):

        em = self.embedLayer(x[0], dicw["dic_"+str(1)])
        for i in range(1,len(x)):
            em = T.concatenate([em,self.embedLayer(x[i], dicw["dic_"+str(i+1)])], axis = em.ndim - 1)
        tens = []
        tens.append(em)
        lay_num = len(self.net_size) - 2
        assert  k <= lay_num

        for i in range(k):
            tens.append(self.applyDropout(self.BLSTMLayer(tens[len(tens)-1], w, "layer_"+str(i+1))))
        return tens[len(tens)-1]

    def cscore(self,x,w,dicw):

        em = self.embedLayer(x[0], dicw["dic_"+str(1)])
        for i in range(1,len(x)):
            em = T.concatenate([em,self.embedLayer(x[i], dicw["dic_"+str(i+1)])], axis = em.ndim - 1)
        self.debug_em = em
        tens = []
        tens.append(em)
        lay_num = len(self.net_size) - 2
        for i in range(lay_num):
            tens.append(self.applyDropout(self.BLSTMLayer(tens[len(tens)-1], w, "layer_"+str(i+1))))

        final_h = tens[len(tens)-1]
        return T.tensordot(final_h, w["wo"], [[final_h.ndim - 1],[0]]) + w["bo"]


    def logp(self,x,y,w,dicw):

        scores = self.cscore(x,w,dicw)
        print "score type  :  "+str(scores.type)

        '''
        final,_ = theano.scan(lambda y_1,y,score,pre_logp,B :pre_logp + T.sum(T.log(T.sum(T.exp(score+T.dot(y_1,B))*y,score.ndim - 1)/ T.sum( T.exp(score+T.dot(y_1,B)), score.ndim - 1))),
                              sequences = [dict(input = y, taps = [-1,0]), scores],
                              outputs_info = np.array(0,dtype = np.float64),
                              non_sequences = w["A"])
        '''
        A = w["A"]
        nu = T.exp(scores+T.tensordot(y[0:y.shape[0]-1],A,1))*y[1:]
        print "numerator type  :  "+str(nu.type)

        de = T.exp(scores+T.tensordot(y[0:y.shape[0]-1],A,1))
        print "denominator  type  :  "+str(de.type)

        final = T.sum(T.log(T.sum(nu, scores.ndim - 1)/T.sum(de, scores.ndim-1)))

        return final



    def beamsearch(self,index_x, bsize, top_n):
        return Model.beamsearch(self,index_x, self.w["A"].get_value(), bsize,top_n)


    def decode(self,index_x, top_n):
        return Model.decode(self,index_x, top_n,self.w["A"].get_value())

    @classmethod
    def create_model(cls, modelfile):

        f_w = open(modelfile, 'rb')

        w = pickle.load(f_w)
        w = dict([(key,w[key].get_value()) for key in w])

        tag_num = w["A"].shape[0]
        net_size = []
        i=0
        while True:
            wlayerid = "wxi"+"layer_"+str(i+1)
            if wlayerid in w:
                net_size.append(w[wlayerid].shape[0])
            else:
                break
            i += 1
        net_size.append(w["wo"].shape[0])
        assert w["wo"].shape[1] == tag_num
        net_size.append(tag_num)

        emb_num = 0
        i=0
        while True:
            wdicid = "dic_"+str(i+1)
            if wdicid in w:
                emb_num += 1
            else:
                break
            i += 1


        return BLSTMmodel(tag_num, emb_num, net_size, premodel = w)
