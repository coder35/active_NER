# coding=utf-8
import theano.tensor as T
import theano
import numpy as np
import pickle
import collections
def shared32(x, name=None, borrow=False):

    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)





class Model:

    def __init__(self, emb_num, wdecay,opt, mweight = 0.9, lrate = 0.01):
        self.emb_num = emb_num
        self.wdecay = wdecay
        self.mweight = shared32(mweight)
        self.lrate = shared32(lrate)
        self.opt = opt
        # these variables are defined by subclass
        self.w = None
    def upda(self,gw,w,lrate , mweight , grad = "momentum", fix_emb = False, eps = 1e-8):
        updates = {}
        assert len(gw) == len(w)

        if grad == "adaGrad":
            for i in range(len(gw)):
                acc = shared32(w[i].get_value()*0.)
                new_acc = acc + gw[i]**2
                if i == len(gw) - 1 and not fix_emb:
                    assert w[i].get_value().shape[0] > 100000
                    assert w[i].get_value().shape[1] == 128
                    print "dictionary update  .............0.1 times"
                    new_w = w[i] - 0.1*lrate*gw[i]/T.sqrt(new_acc + eps)
                else:
                    new_w = w[i] - lrate*gw[i]/T.sqrt(new_acc + eps)

                updates[acc] = new_acc
                updates[w[i]] = new_w


        elif grad == "momentum":

            for i in range(len(gw)):
                mom = shared32(w[i].get_value()*0.)
                new_mom = mom*mweight - lrate*gw[i]

                new_w = w[i] + new_mom

                updates[mom] = new_mom
                updates[w[i]] = new_w

        return collections.OrderedDict(updates)




    def init_w(self):raise NotImplementedError()
    def load_w(self, pre_w, w):
        for key in pre_w:
            assert  key not in w
            w[key] = shared32(pre_w[key])
        return w

    def decode(self,index_x, top_n,trans):
        cs_func = self.cs_func

        road = []
        scores = cs_func(*index_x)


        i = 0
        while i < len(scores):
            score  = scores[i]

            if i == 0:
                node = []
                logp = np.log(np.exp(score)/np.sum(np.exp(score)))

                for k in range(score.shape[0]):
                    node.append([(-1, -1, logp[k])])
                road.append(node)

            else :

                node = []
                logp = []
                for l1 in range(self.tag_num):
                    logp.append(np.log(np.exp(score + trans[l1])/np.sum(np.exp(score + trans[l1]))))

                for l in range(self.tag_num):
                    candidates = []
                    for j in range(len(road[-1])):
                        for k in range(len(road[-1][j])):
                            candidates.append((j,k,road[-1][j][k][2] + logp[j][l]))


                    candidates.sort(lambda x,y:cmp(y[2],x[2]))
                    node.append(candidates[0:top_n])

                road.append(node)
            i += 1
        can = []
        for i in range(len(road[-1])):
            for j in range(len(road[-1][i])):
                can.append((i,j,road[-1][i][j][2]))

        can.sort(lambda x,y:cmp(y[2],x[2]))

        result = []
        for i in range(top_n):
            sequence = []
            tag = can[i][0]
            offset = can[i][1]

            index = len(road) - 1
            try:
                while index >= 0:
                    sequence.append(tag)
                    tag, offset = road[index][tag][offset][0], road[index][tag][offset][1]
                    index  = index - 1
            except Exception:
                print "index  "+str(index)
                print "offset  "+str(offset)
            sequence.reverse()
            result.append((sequence, can[i][2]))


        return  result  # a list of tuples consisted with tage sequence (sequence) and score (exp)s


    def beamsearch(self,index_x, trans, bsize, top_n):
        cs_func = self.cs_func

        scores = cs_func(*index_x) #score顺序为原句顺序的逆序

        road = []
        i = 0
        while i < len(scores):
            score  = scores[i]

            if i == 0:
                node = []
                logp = np.log(np.exp(score)/np.sum(np.exp(score)))

                for k in range(score.shape[0]):
                    node.append((-1, k, logp[k]))

                node.sort(lambda x,y:cmp(y[2], x[2]))

                road.append(node[0:min(len(node), top_n)])

            else :

                node = []
                logp = []
                for j in range(len(road[-1])):
                    nextp = np.log(np.exp(trans[road[-1][j][1]] + score)/np.sum(np.exp(trans[road[-1][j][1]]+score)))
                    for k in range(nextp.shape[0]):
                        node.append((j, k, road[-1][j][2] + nextp[k]))

                node.sort(lambda x,y:cmp(y[2], x[2]))
                road.append(node[0:min(len(node), bsize)])

            i += 1

        result = []
        for i in range(top_n):
            sequence = []
            offset = i
            index = len(road) - 1
            try:
                while index >= 0:
                    tag =  road[index][offset][1]
                    sequence.append(tag)
                    offset = road[index][offset][0]
                    index  = index - 1
            except Exception:
                print "index  "+str(index)
                print "offset  "+str(offset)
            sequence.reverse()
            result.append((sequence, road[-1][i][2]))
        return result

    def testAndPrint(self,indexdata,worddata,tags):
        tagsum = 0.0
        correct = 0.0
        error = 0.0
        iid = 1
        none = tags["NONE"]
        ensum = 0
        for index in range(len(indexdata)):
            item = indexdata[index]
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

            if len(item[0]) != len(worddata[index][0]):
                raise Exception("numdata not match rawdata")

            if len(errorindex) != 0:
                error += 1
                print "error   "+str(iid)
                printsen = zip(worddata[index][0], worddata[index][1],pretags)
                for k in range(len(printsen)):
                    token = printsen[k]
                    if k in errorindex:
                        print token[0].encode('utf-8')+"   "+token[1].encode('utf-8')+"   "+token[2].encode('utf-8')+".........  error "
                    else :
                        print token[0].encode('utf-8')+"   "+token[1].encode('utf-8')+"   "+token[2].encode('utf-8')

                print "          "
        print  "extracted entity  number   :   " + str(ensum)

        return correct/tagsum, error/len(indexdata)


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

    def set_lrate(self,new_lrate):
        self.lrate.set_value(new_lrate)

    def set_mom(self, new_mom):
        self.mweight.set_value(new_mom)


    def printmodel(self,filename):


        f_w = open(filename+"_w", 'wb')
        pickle.dump(self.w, f_w)
        f_w.close()

    def readmodel(self,filename):

        f_w = open(filename+"_w", 'rb')

        self.w = pickle.load(f_w)

