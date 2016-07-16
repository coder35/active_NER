import theano.tensor as T
import theano
import numpy as np
import math
import pickle
import os
import logging
import os.path
import re
def shared32(x, name=None, borrow=False):

    return theano.shared(np.asarray(x, dtype='float64'), name=name, borrow=borrow)





class BASELINEModel(object):

    def __init__(self,dic_size,window,unit_id,tag_num,net_size,weight_decay,word_dim = 50, learning_rate = 0.1):
        def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
        def f_rectlin(x): return x*(x>0)
        def f_rectlin2(x): return x*(x>0) + 0.01 * x
        nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2}
        self.non_unit = nonlinear[unit_id]
        self.weight_decay = weight_decay
        self.tag_num = tag_num
        self.window_size = window
        self.learning_rate = learning_rate
        self.worddim = word_dim
        self.w, self.b, self.A = self.init_w(net_size,tag_num)
        self.w2vtable = self.init_wtable(word_dim,dic_size)#table of word vectors
        x = T.vector('x')
        w = []
        b = []
        for i in range(len(self.w)):
            w.append(T.matrix())
            b.append(T.vector())

        output = self.network(x,w,b)
        og = []
        for j in range(self.tag_num):
            og.extend(T.grad(output[j],w+b+[x]))

        self.outfunction = theano.function([x]+w+b, output)
        self.goutfunction = theano.function([x]+w+b,[output]+og)



    def calGrad(self,x,y):
        if x.shape[0] != y.shape[0]:
            raise Exception("input x not match y")



        ga = np.zeros_like(self.A, dtype = np.float)

        gwb = []
        for i in range(len(self.w)):
            gwb.append(np.zeros_like(self.w[i], dtype = np.float))

        for i in range(len(self.b)):
            gwb.append(np.zeros_like(self.b[i], dtype = np.float))

        gx = np.zeros_like(x, dtype = np.float)

        score = []
        g = []

        for row in x:
            mid = [row] + self.w + self.b
            c = self.goutfunction(*mid)
            score.append(c[0])
            g.append([])
            for i in range(self.tag_num):
                g[-1].append(c[i*(len(gwb)+1) + 1: (i+1)*(len(gwb)+1)+1])


        trans, path_score = self.logadd(score)

	logging.debug(path_score[-1])
        logsum = math.log(sum(path_score[-1])) - self.likeli(y,score)

        gwb = map(lambda x,y:x-y, gwb, g[0][y[0]][0:-1])
        gx[0]  -= g[0][y[0]][-1]


        for j in range(1,y.shape[0]):
            ga[y[j-1]][y[j]] = ga[y[j-1]][y[j]] - 1
            gwb = map(lambda x,y:x - y,gwb, g[j][y[j]][0:-1])
            gx[j] -= g[j][y[j]][-1]



        par = []
        par.append(path_score[-1]/sum(path_score[-1]))
        for k in xrange(self.tag_num):
            gwb = map(lambda x,y :x + par[-1][k]*y, gwb,g[-1][k][0:-1])
            gx[-1] += g[-1][k][-1]*par[-1][k]

        index = y.shape[0]-1

        while index > 0 :
            incre_a = (path_score[index - 1]*(((par[-1]/(np.dot(path_score[index - 1],trans)))*trans).transpose())).transpose()
            ga = ga + incre_a
            par.append(np.sum(incre_a,1))
            for k in xrange(self.tag_num):
                gwb = map(lambda x,y :x + par[-1][k]*y, gwb,g[index - 1][k][0:-1])
                gx[index - 1] += g[index - 1][k][-1]*par[-1][k]

            index = index - 1


        return ga,gwb,gx,logsum

    def index2matrix(self, xindex):
        x = []
        for indexv in xindex:
            vec= []
            for index in indexv:
                vec.extend(self.w2vtable[index])
            x.append(vec)

        return  x

    def upAndEva(self,group):# list of tuples of x and y

        ave_ga = np.zeros_like(self.A)
        ave_gwb = []
        for i in range(len(self.w)):
            ave_gwb.append(np.zeros_like(self.w[i], dtype = np.float))

        for i in range(len(self.b)):
            ave_gwb.append(np.zeros_like(self.b[i], dtype = np.float))

        ave_gx = {}
        group_score = 0
        innum = 0
        for instance in group:
            innum += 1
            index = instance[0]
            x = self.index2matrix(index)
            y = instance[1]

            ga,gwb,gx,logsum = self.calGrad(np.asarray(x,dtype=np.float),np.asarray(y,dtype = np.int))
            group_score += logsum
            ave_ga += ga
            ave_gwb = map(lambda x1,y1:x1+y1, ave_gwb, gwb)

            fnumber = len(index[0])
            for i in range(gx.shape[0]):
                vecs = np.split(gx[i], fnumber)
                for j in range(len(index[i])):
                    ave_gx.setdefault(index[i][j],np.zeros(self.worddim))
                    ave_gx[index[i][j]] += vecs[j]


        ave_ga = ave_ga/float(len(group))
        ave_gwb = map(lambda x: x/float(len(group)), ave_gwb)
        for i in range(len(self.w)):
            ave_gwb[i] += 2*self.weight_decay*self.w[i]

        for key in ave_gx:
            ave_gx[key] /= float(len(group))


	count = 0
	s = 0.
	for item in ave_gwb:
	    s += np.sum(item**2)
	    count += np.size(item)
	print "network weight average gradient  :  "+ str(s/count)
	print "transition matrix average gradient   :"+str(np.sum(ave_ga**2)/np.size(ave_ga))
	count = 0
	s = 0.
	for item in ave_gx:
	    s += np.sum((ave_gx[item])**2)
	    count += np.size(ave_gx[item])
	print "word vector average gradient   :  " +str(s / count)

        self.updateA(ave_ga,self.learning_rate)
        self.updateWB(ave_gwb,self.learning_rate)
        self.updateWord(ave_gx,self.learning_rate)
        return group_score

    def updateA(self,ga,step = 0.1):
        self.A -= ga*step

    def updateWB(self,gwb,step = 0.1):

        if len(gwb) != len(self.w) * 2:
            raise Exception("weight length not match")

        for i in range(len(self.w)):
            adstep = step/self.w[i].shape[0]
            self.w[i] -= adstep * gwb[i]
            self.b[i] -= adstep * gwb[len(self.w)+i]

#        for i in range(len(self.w)):
 #           print self.w[i]



    def updateWord(self,gx,step = 0.1):#gx is a dictionary consisting with (wordindex of self.w2vtable,gradient)  pairs

        for key in gx:
            if key >= len(self.w2vtable):
                raise Exception("word table index out of bound")
            self.w2vtable[key] -= step*gx[key]




    def init_w(self,size,tag_num):
        w = []
        b = []
        for item in size:
            w.append(1./np.sqrt(item[0])*np.random.randn(item[0],item[1]))
            b.append(1./np.sqrt(item[0])*np.random.randn(item[1]))
        A = 0.02 * np.random.randn(tag_num,tag_num)
        return w,b,A

    def init_wtable(self, worddm,size):

        #1(non word at index 0)+ size

        return np.random.randn(size + 1,worddm)


    def logadd(self,score):

        h = np.exp(score)

        trans = np.exp(self.A)

        path_score = []
        path_score.append(np.asarray(h[0],dtype = 'float128'))
        for i in range(1,len(h)):
            path_score.append(np.dot(path_score[-1],trans)*h[i])

        return trans, path_score

    def likeli(self,y,score):
        likelihood = score[0][y[0]]

        for i in range(1,len(score)):
            likelihood += score[i][y[i]] + self.A[y[i-1]][y[i]]

        return likelihood






    def network(self,x,w,b):

        h = x

        for i in range(len(w)):

            if i == len(w) - 1:
                 h = T.dot(h,w[i])+b[i]
            else :
                 h = self.non_unit(T.dot(h,w[i])+b[i])
        return h



    def decode(self,xindex,top_n):
        vectors = self.index2matrix(xindex)
        trans = self.A
        h = []
        for vector in vectors:
            mid = [vector]+self.w+self.b
            h.append(self.outfunction(*mid))

        road = [[]];
        for i in range(self.tag_num):
            road[0].append([(h[0][i],-1,-1)])

        for i in range(1,len(h)):
            road.append([]);
            for i2 in range(self.tag_num):
                candidates = []
                for j in range(len(road[i-1])):
                    for k in range(len(road[i-1][j])):
                        candidates.append((road[i-1][j][k][0] + trans[j][i2] + h[i][i2],j,k))
                candidates.sort(lambda x,y:cmp(y[0],x[0]));
                road[i].append(candidates[0:top_n])

        candidates = []
        for i in range(self.tag_num):
            for j in range(len(road[-1][i])):
                candidates.append((road[-1][i][j][0],i,j));


        candidates.sort(lambda x,y:cmp(y[0],x[0]))
        result = []
        for i in range(top_n):
            sequence = []
            tag = candidates[i][1]
            offset = candidates[i][2]
            index = len(road) - 1
            while index >= 0:
                sequence.append(tag)
                tag, offset= road[index][tag][offset][1], road[index][tag][offset][2]
                index  = index - 1
            sequence.reverse()
            result.append((sequence, candidates[i][0]))




        return  result  # a list of tuples consisted with tage sequence (sequence) and score (exp)s

    def testAndPrint(self,data,rawdata,tags):
        tagsum = 0.0
        correct = 0.0
        error = 0.0
        iid = 1
        none = tags["NONE"]
        ensum = 0
        for index in range(len(data)):
            item = data[index]
            result = self.decode(item[0],1)
            if len(item[1]) != len(result[0][0]):
                raise Exception("test error")

            errorindex = []
            pretags = []
            for i in range(len(item[1])):
                if item[1][i] == result[0][0][i]:
                    correct +=1
                else :
                    errorindex.append(i)
                if result[0][0][i] != none:
                    ensum += 1

                for key in tags:
                    if tags[key] == result[0][0][i]:
                        pretags.append(key)
                        break

                tagsum+=1

            if len(item[0]) != len(rawdata[index][0]):
                raise Exception("numdata not match rawdata")

            if len(errorindex) != 0:
                error += 1
                print "error   "+str(iid)
                printsen = zip(rawdata[index][0], rawdata[index][1],pretags)
                for k in range(len(printsen)):
                    token = printsen[k]
                    if k in errorindex:
                        print token[0].encode('utf-8')+"   "+token[1].encode('utf-8')+"   "+token[2].encode('utf-8')+".........  error "
                    else :
                        print token[0].encode('utf-8')+"   "+token[1].encode('utf-8')+"   "+token[2].encode('utf-8')

                print "          "
        print  "extracted entity  number   :   " + str(ensum)

        return correct/tagsum, error/len(data)


    def test(self,data):

        tagsum = 0.0
        correct = 0.0
        for item in data:
            result = self.decode(item[0],1)
            if len(item[1]) != len(result[0][0]):
                raise Exception("test error")

            for i in range(len(item[1])):
                if item[1][i] == result[0][0][i]:
                    correct +=1
                tagsum+=1

        return correct/tagsum



    def printmodel(self,filename):


        f_w = open(filename+"_w", 'wb')
        f_b = open(filename+"_b", 'wb')
        f_A = open(filename+"_A", 'wb')
        f_dic = open(filename+"_dic", 'wb')

        pickle.dump(self.w, f_w)
        pickle.dump(self.b, f_b)
        pickle.dump(self.A, f_A)
        pickle.dump(self.w2vtable, f_dic)

        f_w.close()
        f_b.close()
        f_A.close()
        f_dic.close()

    def readmodel(self,filename):

        f_w = open(filename+"_w", 'rb')
        f_b = open(filename+"_b", 'rb')
        f_A = open(filename+"_A", 'rb')
        f_dic = open(filename+"_dic", 'rb')

        self.w = pickle.load(f_w)
        self.b = pickle.load(f_b)
        self.A = pickle.load(f_A)
        self.w2vtable = pickle.load(f_dic)




    @classmethod
    def get_bestmodel(cls, targetdir,tag_num):
        bestscore = 0
        t =  ""
        for parent, dirnames, filenames in os.walk(targetdir):

            if parent.endswith(targetdir):
                for dname in dirnames:
                    if dname.startswith("hid"):
                        for p, d, f in os.walk(os.path.join(parent,dname)):
                            for item in f:
                                if item == "score":
                                    score = open(os.path.join(parent,dname,item))
                                    m = re.findall(r'(\D+)(\d+)(\D+)(\d\.\d+)', score.readline())
                                    for tu in m:
                                        if float(tu[3]) > bestscore:
                                            t = os.path.join(parent, dname, "model_"+tu[1]+"_")
                                            bestscore = float(tu[3])


        f = open(t+"w")
        w = pickle.load(f)
        f= open(t+"b")
        b = pickle.load(f)
        f=open(t+"A")
        A = pickle.load(f)
        f= open(t+"dic")
        w2vtable = pickle.load(f)

        net_size = []
        for item in w:
            net_size.append((item.shape[0],item.shape[1]))
        model = BASELINEModel(0,0,'tanh',tag_num,net_size,0)

        model.w = w
        model.b = b
        model.A = A
        model.w2vtable = w2vtable


        return model

