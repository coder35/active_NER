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
from time import gmtime, strftime
#@profile
def getsemidata(filename, cdic,negnum = 0):
    #get partial labeld data : a list of  tuples ,each tuple contains char index sequence  and entiuies ,like   ([23,1,3,434,231,5,65],[[0,3],[5,7]])
    ndata = []
    for i in range(3):
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
        cdata[i] = [inseq] + item[1:]
    return
def token2index2(data,indic,outdic):
    for index,item in enumerate(data):
        inseq = []
        outseq = []
        for token in item[0]:
            inseq.append(indic.get(token,0))
            outseq.append(outdic.get(token,0))
        data[index] = [inseq]+item[1:]+[outseq]
    return
def wiki2tdata(filename, indic, maxnum,negnum = 0, entitymask = 0,ismask = False, maskpro = 0.2, isLM = False, outdic = None):
    #file 中每一项为[ sentence, [entity_start, entity_end], entity types]  like[[乔,布,斯,是,美,国,人],[0,3],[person.person]]
    wiki = open(filename)

    data = []
    clusters = {}
    types = {}
    num = 0
    for line in wiki:
        item = json.loads(line)
        #if len(data) == 1000:
        #    break
        if len(data)%10000  == 0:
            print "read wiki data  number :  "+str(len(data))
        if maxnum > 0 and len(data) > maxnum:
            break
        if len(item[0]) > 20 and len(item[0]) < 100:
            typeids = []
            for entype in item[2]:
                if entype not in types:
                    types[entype] = len(types)
                typeids.append(types[entype])

           

            data.append([item[0],item[1],typeids])
            #test
            '''
            if num == 3:
                break
            num += 1
            '''
    if isLM:
        token2index2(data, indic,outdic)
        assert len(data[0]) == 4

    else:
        char2index(data,indic)
        assert len(data[0]) == 3
    print data[0][0][data[0][1][0]:data[0][1][1]]
    dic = None
    if negnum > 0:
	dic  = {}
        for item in data:
            dic.setdefault(tuple(item[0]), set())
	    #print item[1]
            for index in range(item[1][0], item[1][1]):
                dic[tuple(item[0])].add(index)
    for item in data:
	item.append(dic[tuple(item[0])])
    
    if ismask:
    	print "adopt entity masking stratergy..........mask probability : "+str(maskpro)

        for item in data:
            ran = random.random()
            if ran < maskpro:
                for index in range(item[1][0], item[1][1]):
                    item[0][index] = entitymask

    
    print data[0][0][data[0][1][0]:data[0][1][1]]

    traindata = gentraindata(data, types, negnum)
    return traindata, types


def gentraindata(indexdata, types,negnum = 0, wsize=0):
    #indexdata   each item like  ([2,4,2,4,5],[1,2],[1,2,3])
    #negnum  :每个句子产生negnum个负例
    #wsize :输入x截取的窗口大小
    #enseq,nonseq  each item like ([ [[2],[3],[1],[6]] ], [0,1,1,0],[-1,1,1,1])
    enseq = []
    nonenseq = []

    for item in indexdata:
        charseq = reshapex(item[0],wsize)
        en = item[1]

        assert en[0] < en[1]
        y = np.asarray([0]*en[0]+[1]*(en[1] - en[0])+[0]*(len(charseq) -  en[1]), dtype = np.int16)
        label = np.asarray(range(len(types)),dtype = np.int32 )
        for index in xrange(len(label)):
            if index in item[2]:
                label[index] = 1
            else:
                label[index] = -1

        enseq.append([[ charseq ], y, label])

        for j in xrange(negnum):
            flag = False
            while not flag:
                flag = True
                sindex = random.randint(0, len(charseq) - 1)
                en = item[3]
                #print en
		if sindex in en:
                    flag = False

                if flag:
		    #print sindex
                    y = np.asarray([0]*sindex + [1] + [0]*(len(charseq)-sindex-1), dtype = np.int16)

            nonenseq.append([[charseq ], y, np.asarray(len(types)*[-1], dtype = np.int16)] )


    print "entity data number stat .............. "
    print "wiki positive  instance  number  :  "+str(len(enseq))+" negative  instances number  :  "+str(len(nonenseq))
    #print enseq[0] 
    #print "............."
    #print nonenseq[0]

    return enseq+nonenseq

def genAnEpoch(endata, group_size,group_num):
    endata = randomlist(endata)

    dic = {}
    for i in xrange(20,100,10):
        dic[i] = []

    for item in endata:
        #print len(item[1])
        dic[len(item[1])/10*10].append(item)

    groups = []
    for key in dic:
        groups.extend([dic[key][start:start + group_size] for start in xrange(0,len(dic[key]) - group_size, group_size)])
    if group_num == -1:
        groups = randomlist(groups)
    else:
        groups = randomlist(groups)[:group_num]

    return groups


def randomlist(ilist):
    for i in xrange(len(ilist)):
        rindex = random.randint(i, len(ilist)-1)
        mid = ilist[rindex]
        ilist[rindex] = ilist[i]
        ilist[i] = mid

    return ilist

def padgroup(group,padding_id):
    maxlen = max([len(item[0][0]) for item in group])
    #print "max length  :  "+str(maxlen)

    def padinstance(instance):
        assert len(instance[0][0]) == len(instance[1])
        #print instance[0][0]
        newx = np.pad(instance[0][0], ((0,maxlen - len(instance[0][0])), (0,0)), 'constant', constant_values = (padding_id, padding_id))
        newy = np.pad(instance[1], (0,maxlen - len(instance[1])), 'constant', constant_values = (0,0))
        return [[newx],newy]+instance[2:]
    return map(padinstance, group)

def padgroup2(group,padding_id):
    maxlen = max([len(item[3]) for item in group])
    assert len(group[0][0][0]) == maxlen

    def padinstance(instance):
        newout = np.pad(instance[3],(1,maxlen+1-len(instance[3])),'constant', constant_values = (padding_id,padding_id))
        return instance[:-1]+[newout]
    return map(padinstance, group)


def reshapex(listx,wsize):
    newx = [0]*wsize + listx + [0]*wsize
    indexma = [newx[current-wsize:current+wsize+1] for current in xrange(wsize, len(newx) - wsize)]
    return np.asarray(indexma, dtype = np.int32)


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



def writedataAnddic(dirname,indexdata, tokendic, tagdic, endic = None):
    nerstr = dirname
    if os.path.exists(nerstr):
        shutil.rmtree(nerstr)
    os.mkdir(nerstr)
    af = open(nerstr+"/data", 'wb')
    pickle.dump(indexdata,af)
    af.close()

    df = open(nerstr+"/tokendic", 'wb')
    pickle.dump(tokendic,df)
    df.close()

    tf = open(nerstr+"/tags",'wb')
    pickle.dump(tagdic,tf)
    tf.close()

    if endic:
        ef = open(nerstr+"/entypes",'wb')
        pickle.dump(endic,ef)
        ef.close()


def testmodel(testdata,sfilename,epoch,model,top_n,tags):
    predict = []
    gold = []
    print  "start  decoding   .............."
    for item in testdata:
        predict.append(map(lambda x:x[0], model.decode(item[0], top_n)))
        gold.append(item[1])
    print  "decoding   finished  ..........."
    pr_dic = pr.computePR(tags, gold, predict)
    print  "test  result   :   "

    rev  = dict(zip(tags.values(), tags.keys()))
    
    score_file = open(sfilename,'ab')
    for key in pr_dic:
        precision = pr_dic[key][0]
        recall = pr_dic[key][1]
        f = 2./(1./(precision+1e-7)+1./(recall+1e-7))
        if key == "overall":
            overallf = f
            entype = "overall"
        else:
            entype = rev[key]
    
        s = "epoch  :  "+str(epoch)+"   entity  type  :  "+entype+"    p :  " +str(precision)+"  R   :   "+str(recall) + "   F  :  "+str(f)+"\n"
        print s
        score_file.write(s)
        
    score_file.close()
    return overallf





dim = 128
dropout_pro = 0.3

storedir = sys.argv[1]
batch_size = 10
em_num = 1
top_n = 1

layerid = 2
wgroup_size = 60#每个batch 包含 wgroup_size 个instances
pretime = 5
'''
#get sighan ner data
traindata = preprocess.get_sighan_wdata("sighanNER/traindata")
devdata = preprocess.get_sighan_wdata("sighanNER/devdata")
testdata = preprocess.get_sighan_wdata("sighanNER/testdata")
tags = preprocess.tags2dic(map(lambda x:x[1], traindata))

print "loading  word vectors ......................."
wdic, wvectors = preprocess.words2dic2("baiduvector_20", dim)
print "word vectors loading is done .............vector number "+str(len(wvectors))


padding_id = wdic["<padding>"] = len(wvectors)
pids = [padding_id]
wvectors.append(np.random.randn(dim))
embedding = wvectors
traindata = preprocess.raw2num1(traindata,wdic,tags,0,padding_id)
devdata = preprocess.raw2num1(devdata, wdic, tags,0,padding_id)
testdata = preprocess.raw2num1(testdata, wdic, tags,0,padding_id)
indexdata = traindata+devdata+testdata
'''
#get bosen labeled data

data = preprocess.getwData("data.txt")
tags = preprocess.tags2dic(map(lambda x:x[1], data))


# start pre-training  from scratch
print "loading  word vectors ......................."
wdic, wvectors = preprocess.words2dic2("baiduvector_20", dim)
print "word vectors loading is done .............vector number "+str(len(wvectors))
padding_id = wdic["<padding>"] = len(wvectors)
pids = [padding_id]
wvectors.append(np.random.randn(dim))
embedding = wvectors
embeddic = {"dic_1":embedding}
premodel = None


#load existing model to continue pre-training 
'''
premodel = pickle.load(open("pretrain_mask/premodel_epoch1_w"))
embeddic = pickle.load(open("pretrain_mask/premodel_epoch1_dic"))
wdic = pickle.load(open("pretrain_mask/tokendic"))
print "word vectors loading is done .............vector number "+str(len(embeddic["dic_1"]))
padding_id = wdic["<padding>"]
pids = [padding_id]
'''


indexdata = preprocess.raw2num1(data,wdic,tags,0,padding_id)
traindata = indexdata[0:len(indexdata)/20*16]
devdata = indexdata[len(indexdata)/20*16:len(indexdata)/20*18]
testdata = indexdata[len(indexdata)/20*18:len(indexdata)]


#semisupervised data

isLM = False
pretrain = True 
if isLM and pretrain:
    outdic = json.load(open("words_50000.json"))
    outpadding_id = len(outdic)
    outdic["<padding>"] = outpadding_id
    wordnum = len(outdic)
    print "language model  target words type number  :  "+str(wordnum)
    mtype = "multask"
    LMweight = 0.02
    print " get  wiki data....................................."
    endata, entypedic = wiki2tdata('newendata.json', wdic, -1, entitymask = 0,ismask = True, isLM = isLM,outdic = outdic)
    writedataAnddic(storedir, indexdata, wdic, tags, entypedic)
    labelnum = len(entypedic)
elif pretrain:
    outdic = None
    mtype = "mullabel"
    wordnum = 0
    LMweight = 0
    outpadding_id = 0
    print " get  wiki data....................................."
    entitymask = "<entity>"
    
    #wdic["entitymask"] = len(embedding)
    #embedding.append(np.random.randn(dim))
    
    endata, entypedic = wiki2tdata('endata.json', wdic, -1,negnum = 2, entitymask = 0,ismask = True,maskpro = 1,isLM = isLM,outdic = outdic)
    
    writedataAnddic(storedir, indexdata, wdic, tags, entypedic)
    labelnum = len(entypedic)
else:
    mtype = "supervised"
    outdic = None
    wordnum = 0
    LMweight = 0
    outpadding_id = 0
    writedataAnddic(storedir, indexdata, wdic, tags, None)
    labelnum = 0



print "writing  data  is  done .................................."



#list1 = range(100,110,10)#tune
#list2 = [0]*len(list1)#tune
#list1 = [100]*10
#list2 = range(5,55,5)
#for hidden_num, semiweight in zip(list1,list2):
list1 = [120]
list2 = [90]
for hidden_num1, hidden_num2 in zip(list1,list2):
        semiweight = 0
        #semiweight = semiweight/10.
        #net_size = [dim, hidden_num1*2,len(tags)]
        net_size = [dim, hidden_num1*2,hidden_num2*2,len(tags)]
        print "model initialization ..............model type  "+str(mtype)
        print "model  initialization ..............network size  "+str(net_size)
        print "model  initialization .............pre-training  epoch  "+str(pretime)

        #model = semimodel(len(tags),em_num, net_size,typenum = labelnum, model_type = mtype, wordnum = wordnum, LMweight = LMweight, padding_id = outpadding_id,layerid = layerid,
        #dropout = dropout_pro, lrate=0.03, wdecay = 0., opt = "adaGrad",fix_emb = False, embeddic = {"dic_1": embedding})
         
        model = semimodel(len(tags),em_num, net_size,typenum = labelnum, model_type = mtype, wordnum = wordnum, LMweight = LMweight, padding_id = outpadding_id,layerid = layerid,
        dropout = dropout_pro, lrate=0.02, wdecay = 0., opt = "adaGrad",fix_emb = False, embeddic = embeddic, premodel = premodel)


        print "model training preparing.........................."
        model.train_ready()
        model.evaluate_ready()
        model.unsup_evaluate_ready()
        print "model training ready ............................."
        modelstr = storedir+"/hid1_"+str(hidden_num1)+"_hid2_"+str(hidden_num2)
        if os.path.exists(modelstr):
            shutil.rmtree(modelstr)
        os.mkdir(modelstr)
        os.mknod(modelstr+"/trainscore")
        os.mknod(modelstr+"/devscore")
	
        for i in xrange(0,pretime):
            wikibatches = genAnEpoch(endata, wgroup_size,-1)
            print  " epoch  group  number   ................................."+str(len(wikibatches))
            wikibatches = [padgroup(group,padding_id) for group in wikibatches]
            if isLM:
                wikibatches = [padgroup2(group,outpadding_id) for group in wikibatches]
            #for item in wikibatches[0]:
            #    print item[3]

            #write out entity data
            conf = open(modelstr+"/endata.con",'w')
            pickle.dump(endata,conf)
            conf.close()
            endata = None

            num = 0
            trainwiki = wikibatches[:len(wikibatches) - 50]
            testwiki = wikibatches[len(wikibatches) - 50 :]
            print "............................ pre-train  time   "+str(i)
            for batch in trainwiki:
                num += 1
                if num % 2000 == 0:
                    correct = 0.
                    model.set_dropout(0.)
                    for batch in testwiki:
                        correct += model.unsup_evaluate(batch)
                    model.set_dropout(dropout_pro)
                    wikime = "wiki training  precision  :  "+str(correct/(50*wgroup_size*len(entypedic))) +"   "+strftime("%Y-%m-%d %H:%M:%S", gmtime())+"\n"
                    print wikime
                    fi = open(modelstr+"/wikiprecision",'ab')
                    fi.write(wikime)
                    fi.close()

                sgroupscore = model.unsuperUpdate(batch)
                print "score   :  "+str(sgroupscore)+"................batch  id  :  "+str(num)
		
		if num == len(trainwiki)/2:
		    model.printmodel(modelstr+"/premodel_epoch"+str(i)+"_half", True)
            model.printmodel(modelstr+"/premodel_epoch"+str(i+1), True)

            #read in entity data
            conf = open(modelstr+"/endata.con")
            endata = pickle.load(conf)

            conf.close()

        model.set_lrate(0.10)
        print "start fine-tuning   .........."
        
	epoch  = 1
        bestf = 0.
        while True:
            print "epoch  .................................................................."+str(epoch)
            traindata = preprocess.randomdata(traindata)
            #随机打乱数据
            print "generate batches ......."
            supergroups = preprocess.data2batch(traindata,batch_size,pids)
            print "start  joint update .............."
            semiscore = 0
            #wikibatches = [padgroup(group,padding_id) for group in genAnEpoch(endata,wgroup_size,len(supergroups))]
            for sgroup in supergroups:
                #groupscore = model.semiUpdate(wikigroup, sgroup)
                groupscore = model.superUpdate(sgroup)

                print "group loss : "+str(groupscore)
                semiscore+=groupscore

            print " epoch   all   loss :  "+str(semiscore)
            print "update over  ....."

            #test on dev data
            print "...............test on development data "
            sfile = modelstr+"/devscore"
            model.set_dropout(0.)
            f = testmodel(devdata, sfile, epoch,model,top_n,tags)
            model.set_dropout(dropout_pro)

            if f > bestf and epoch > 3:
                model.printmodel(modelstr+"/model_epoch"+str(epoch),False)
                bestf = f
                sfile = modelstr+"/testscore"
                model.set_dropout(0.)
                testmodel(testdata, sfile, epoch, model,top_n,tags)
                model.set_dropout(dropout_pro)

            if epoch == 5 or epoch == 11:
                sfile = modelstr+"/trainscore"
                model.set_dropout(0.)
                testmodel(traindata, sfile, epoch,model,top_n,tags)
                model.set_dropout(dropout_pro)

            if epoch == 6:
                model.set_lrate(0.03)
            elif epoch == 10:
                model.set_lrate(0.015)
            elif epoch == 13:
                break
            epoch += 1

