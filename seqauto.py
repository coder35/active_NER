import BLSTM

class autoencoder(BLSTMmodel):

    def __init__(self, tag_num,emb_num, net_size,drop_pro = 0.6, mweight = 0.9, lrate = 0.01, opt = "momentum", wdecay = 0., embeddic = None, premodel =None):
        Model.__init__(self,emb_num, wdecay,opt, mweight, lrate)
        def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
        def f_rectlin(x): return x*(x>0)
        def f_rectlin2(x): return x*(x>0) + 0.01 * x
        self.nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2}
        self.tag_num = tag_num
        self.net_size = net_size

        self.wdecay = wdecay

        self.w = self.init_w(embeddic, premodel)

        self.dropout_prob = shared32(drop_pro)
        self.srng = T.shared_randomstreams.RandomStreams(random.randint(0,9999))



    def raw2input(self,group):
        #堆积group中每个实例，将y的索引表示转化为one-hot 向量表示，产生theano函数符合条件的输入
        groups_x = []
        group_y = []
        group_index = []

        selen = len(group[0][1]) #sequence length
        embnum = len(group[0][0])#number of differnet index group in x
        for i in range(embnum):
            groups_x.append([])
            for j in range(selen):
                groups_x[-1].append([])
        for i in range(selen + 1):# the length of y is equal to that of x plus 1, the first one in y is used as start symbol
            group_y.append([])

        print "sequence length   :   "+str(selen)
        for instance in group:# for each instance  eg. instance[0]:([[1,2],[2,3]],[[1],[2]])  instance[1] : [3 ,-1] 小于0代表padding的标签
            for i in range(len(instance[0])):
                indexgroup = instance[0][i]
                for j in range(selen):
                    groups_x[i][j].append(indexgroup[j])



            y = np.arange(self.tag_num*(selen + 1), dtype = np.int32).reshape((selen + 1,self.tag_num))*0

            for i in range(1, selen + 1):
                if instance[1][i-1] < 0:
                    for j in range(len(y[i])):
                        y[i][j] = 1

                else:
                    y[i][instance[1][i-1]] = 1

            for i in range(selen + 1):
                group_y[i].append(y[i])



        funcinput = [np.asarray(x, dtype = np.int32) for x in groups_x] + [np.asarray(group_y, dtype = np.int32)]
        return funcinput


    def init_w(self, embdic = None, premodel = None):
        w = {}
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

            w["wo"] = shared32(1./np.sqrt(size[-2])*np.random.randn(size[-2], size[-1]))
            w["bo"] = shared32(np.random.randn(size[-1]))

            if embdic == None:
                raise Exception("embdedding dic not initialization")
            else:
                for key in embdic:
                    w[key] = shared32(embdic[key])


        else :
            for key in premodel:
                w[key] = shared32(premodel[key])
            w["A"] = shared32(np.random.randn(n_t,n_t))

            w["wo"] = shared32(1./np.sqrt(size[-2])*np.random.randn(size[-2], size[-1]))
            w["bo"] = shared32(np.random.randn(size[-1]))


        return w




    def logp(self,x,y,w):
        scores = self.cscore(x,w)
        '''
        final,_ = theano.scan(lambda y_1,y,score,pre_logp,B :pre_logp + T.sum(T.log(T.sum(T.exp(score+T.dot(y_1,B))*y,score.ndim - 1)/ T.sum( T.exp(score+T.dot(y_1,B)), score.ndim - 1))),
                              sequences = [dict(input = y, taps = [-1,0]), scores],
                              outputs_info = np.array(0,dtype = np.float64),
                              non_sequences = w["A"])
        '''
        A = w["A"]
        final = T.sum(T.log(T.sum(T.exp(scores+T.tensordot(y[0:-1],A,1))*y[1:],scores.ndim - 1)/T.sum(T.exp(scores+T.tensordot(y[0:-1],A,1)),scores.ndim - 1)))

        return final




