# encoding=utf-8
import json
import random
import numpy as np
import preprocess
from semiBLSTM import semimodel
import os
import pickle
import shutil
import pr
import util
#from memory_profiler import  profile
import sys
import gc
#@profile
def getsemidata(filename, cdic,negnum = 0):
    #get partial labeld data : a list of  tuples ,each tuple contains char index sequence  and entiuies ,like   ([23,1,3,434,231,5,65],[[0,3],[5,7]])
    ndata = []
    for i in range(5):
        data = json.load(open(filename+"-"+str(i).zfill(5)+".json"))
        for item in data:
            if len(item[0]) > 20 and len(item[0]) < 100:
                ndata.append(item)

    char2index(ndata, cdic)
    return gentraindata(ndata, negnum)



def char2index(cdata, cdic):
    # input :a list of tuple (each tuple like ([乔,布,斯,创,立,苹,果],[[0,3],[5,7]])
    # output :a list of tuple (each tuple like ([23,1,3,434,231,5,65],[[0,3],[5,7]])
    for i in xrange(len(cdata)):
        item = cdata[i]

        inseq = []
        for char in item[0]:
            cdic.setdefault(char,0)
            inseq.append(cdic[char])
        cdata[i] = (inseq,item[1])

    return



def gentraindata(indexdata, negnum, wsize=0):
    #indexdata   each item like  ([2,4,2,4,5],[[1,2],[2,4]])
    #negnum  :每个句子产生negnum个负例
    #wsize :输入x截取的窗口大小
    #enseq,nonseq  each item like ([ [[2],[3],[1],[6]] ], [0,1,1,0])
    enseq = []
    nonenseq = []
    for item in indexdata:
        charseq = reshapex(item[0],wsize)
        ens = item[1]
        for en in ens:
            assert en[0] < en[1]
            y = np.asarray([0]*en[0]+[1]*(en[1] - en[0])+[0]*(len(charseq) -  en[1]), dtype = np.int16)
            enseq.append([[ charseq ], y])

        for j in xrange(negnum):
            flag = False
            while not flag:
                flag = True
                seglen = random.randint(1,5)
                sindex = random.randint(0, len(charseq) - segleni)
                for en in ens:
                    if sindex>= en[0] and sindex+seglen<=en[1]:
                        flag = False
                        break
                if flag:
                    y = np.asarray([0]*sindex + [1]*seglen + [0]*(len(charseq)-sindex-seglen), dtype = np.int16)

            nonenseq.append(([charseq ], y))


    return enseq, nonenseq

def padgroup(group,padding_id):
    maxlen = max([len(item[0][0]) for item in group])
    def padinstance(instance):
        assert len(instance[0][0]) == len(instance[1])
        newx = np.pad(instance[0][0], ((0,maxlen - len(instance[0][0])), (0,0)), 'constant', constant_values = (padding_id, padding_id))
        newy = np.pad(instance[1], (0,maxlen - len(instance[1])), 'constant', constant_values = (0,0))
        return ([newx],newy)
    return map(padinstance, group)

def reshapex(listx,wsize):
    newx = [0]*wsize + listx + [0]*wsize
    indexma = [newx[current-wsize:current+wsize+1] for current in xrange(wsize, len(newx) - wsize)]
    return np.asarray(indexma, dtype = np.int16)


#@profile
def embeddata(endata,model,padding_id):
    #过滤掉长度小于15 且大于150的item， 并得到data的embedding
    fields = [(20,25),(25,40),(40,60),(60,100)]

    groups = []
    for i in range(len(fields)):
        groups.append([])

    for item in endata:
        for index in range(len(fields)):
            if len(item[0][0]) >= fields[index][0] and len(item[0][0]) < fields[index][1]:
                groups[index].append(item)
                break

    while len(endata) != 0:
        del endata[0]
    newendata = []
    for group in groups:
        newendata += group

    af = open("con_endata", 'wb')
    print "start writing temporary  entity data..................."
    pickle.dump(newendata,af)
    af.close()


    while len(newendata) != 0:
        del newendata[0]

    embeddings = model.computeEmb(padgroup(groups[0], padding_id))
    groups[0] = None
    print "garbage collecting starts ..............."
    gc.collect()
    print "garbage collecting ends ..............."

    for i in xrange(1, len(groups)):
        embeddings = np.vstack([embeddings, model.computeEmb(padgroup(groups[i], padding_id))])
        groups[i] = None
        print "garbage collecting starts ..............."
        gc.collect()
        print "garbage collecting ends ..............."

    return embeddings



def gen_clusters(clusters, endata,cnum,inum):
    cid = range(len(clusters))

    groups = []
    for i in xrange(cnum):
        cluster = []
        while len(cluster) == 0:
            cindex = random.randint(0,len(cid)-1)
            cluster = clusters[cid[cindex]]

        groups.append([])
        for j in xrange(inum):
            groups[-1].append(endata[cluster[random.randint(0,len(cluster)-1)]])
        cid.remove(cid[cindex])

    return groups



def writedataAnddic(dirname,indexdata, tokendic, tagdic):
    nerstr = dirname
    if os.path.exists(nerstr):
        shutil.rmtree(nerstr)
    os.mkdir(nerstr)
    af = open(nerstr+"/data", 'wb')
    pickle.dump(indexdata,af)
    af.close()

    df = open(nerstr+"/chardic", 'wb')
    pickle.dump(tokendic,df)
    df.close()

    tf = open(nerstr+"/tags",'wb')
    pickle.dump(tagdic,tf)
    tf.close()



chardim = 50
cdic, cvectors = preprocess.chars2dic2("char_vector_50",chardim)
storedir = "semimodel2"
batch_size = 20

em_num = 1
top_n = 1
#get labeled data
data = preprocess.getcData("data.txt")
tags = preprocess.tags2dic(map(lambda x:x[1], data))
padding_id = cdic["<padding>"] = len(cvectors)
pids = [padding_id]
cvectors.append(np.random.randn(chardim))
embedding = cvectors
indexdata = preprocess.raw2num1(data,cdic,tags,0,padding_id)
traindata = indexdata[0:len(data)/20*16]
devdata = indexdata[len(data)/20*16:len(data)/20*18]
testdata = indexdata[len(data)/20*18:len(data)]

writedataAnddic(storedir, indexdata, cdic, tags)


'''
fname = "entitydata"
print " start write processed entity data ..............."
f = open(fname,'w+')
pickle.dump(endata,f)
f.close()
sys.exit()
'''
#semisupervised weight

layerid = 2
cnum = 30 #对百度百科含有实体数据总共聚类的数目
sam_cnum = 3#每次取样所取的类别的数目
sam_inum = 20 #每个聚类所取实例个数
hinge = 1.
cluster_point = 5

for hidden_num in range(64,320,32):
    for magweight in range(6,20,2):
        magweight = magweight/10.

        # drop_pro = float(drop)/10.
        net_size = [chardim, hidden_num*2, hidden_num, len(tags)]
        #net_size = [chardim, hidden_num*2, hidden_num,len(tags)]
        print "model  initialization ............network size  "+str(net_size)
        model = semimodel(len(tags),em_num, net_size,magweight, sam_cnum, layerid, hinge, lrate=0.02, opt = "adaGrad", embeddic = {"dic_1": embedding})
        print "model training preparing.........................."
        model.train_ready(True)
        model.evaluate_ready()
        print "model training ready ............................."
        epoch  = 1
        modelstr = storedir+"/hid_"+str(hidden_num)+"_magweight_"+str(magweight)
        if os.path.exists(modelstr):
            shutil.rmtree(modelstr)
        os.mkdir(modelstr)
        #os.mknod(modelstr+"/trainscore")
        os.mknod(modelstr+"/devscore")
        while True:
            print "epoch  .................................................................."+str(epoch)
            traindata = preprocess.randomdata(traindata)
            #随机打乱数据
            print "generate batches ......."

            if epoch >= cluster_point:
                if epoch == cluster_point:
                    #get partial labeled data
                    print " start getting baidu entity data ................"
                    endata, nonendata = getsemidata("semidata/endata", cdic)

                    print  "start  embedding data ............... "
                    embeddings=  embeddata(endata,model,padding_id)

                    print  "embedding data done ............... embedding size :  "+str(len(embeddings[0]))

                    print "  start clustering ........................."
                    clusters = util.k_means(embeddings,cnum)
                    for key in clusters:
                        print str(key)+"  :  "+str(len(clusters[key]))
                    del embeddings
                    print "collect  embeddings garbage..................."

                    gc.collect()
                    af = open("con_endata", 'rb')
                    print "start reading temporary entity data..................."
                    endata = pickle.load(af)
                    af.close()

                supergroups = preprocess.data2batch(traindata,batch_size,pids)

                print "start  joint update .............."
                semiscore = 0
                count = 0
                for sgroup in supergroups:
                    #model.train_ready(group)
                    mclusters = [padgroup(group, padding_id) for group in gen_clusters(clusters, endata, sam_cnum, sam_inum)]
                    sgroupscore = model.semiUpdate(mclusters, sgroup)
                    overalloss = sgroupscore[0]
                    #clusterloss = sgroupscore[1:]
                    print "group loss : "+str(overalloss)
                    semiscore+=overalloss
                print " epoch   all   loss :  "+str(semiscore)

            else:

                supergroups = preprocess.data2batch(traindata,batch_size,pids)

                #debug
                #supergroups = supergroups[0:5]
                print "start  supervision update .............."
                for sgroup in supergroups:
                    #model.train_ready(group)
                    sgroupscore = model.superUpdate(sgroup)
                    print "group loss : "+str(sgroupscore)


            print "update over  ....."





            #test on dev data
           
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
            s = "epoch  :  "+str(epoch)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
            print s
            score_file.write(s)

            #test on train data
            '''
            score = model.test(traindata)
            predict = []
            gold = []
            for item in traindata:
                predict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
                gold.append(item[1])


            precision, recall = pr.computePR(tags, gold, predict)
            f = 2./(1./precision+1./recall)
            score_file = open(modelstr+"/trainscore",'ab')
            print  "test result   :   "
            s = "epoch  :  "+str(epoch)+"   tag   score  :  "+str(score)+" P:  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
            print s
            score_file.write(s)
            '''
            score_file.close()



            if epoch == 25:
                model.set_lrate(0.01)
            elif epoch == 30:
                break
            epoch += 1





