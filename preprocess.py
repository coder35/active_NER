# encoding=utf-8
import os
import jieba
import io
import re
import random
import sys
import numpy as np
def splisen(sen):


    dot = io.open("dot.txt", encoding = 'utf-8')
    sp = dot.readline()[0]

    sens = sen.split(sp)
    i = 0
    while i < len(sens):
        newsen = sens[i].strip()
        if len(newsen) == 0:
            sens.pop(i)
        else:
            sens[i] = newsen
            i += 1
    return sens



def getcData(filename):

    wdata = getwData(filename)

    data = []

    for words, wtags in wdata:
        chars = []
        ctags = []
        for word, tag in zip(words, wtags):
            chars.extend([char for char in word])
            if tag == "NONE":
                ctags.extend(["NONE" for i in range(len(word))])
            elif tag.startswith("B-"):
                ctags.append(tag)
                for i in range(1,len(word)):
                    ctags.append("I-"+tag[2:])
            elif tag.startswith("I-"):
                for i in range(len(word)):
                    ctags.append(tag)
            elif tag.startswith("E-"):
                for i in range(1,len(word)):
                    ctags.append("I-"+tag[2:])
                ctags.append(tag)
            elif tag.startswith("S-"):
                if len(word) == 1:
                    ctags.append(tag)
                else:
                    ctags.append("B-"+tag[2:])
                    for i in range(2,len(word)):
                        ctags.append("I-"+tag[2:])
                    ctags.append("E-"+tag[2:])


        data.append((chars, ctags))


    return data


'''
data = getcData("data.txt")

for i in xrange(len(data[0][0])):
    print data[0][0][i].encode("utf-8")+" "+data[0][1][i].encode("utf-8")
'''
def get_weibo_cdata(filename):

    data = io.open(filename, encoding = 'utf-8')

    sentence = []
    entities = []
    offset = 0
    state = 0
    newdata = []
    for line in data:
        if line == "\n":
            if state:
                entities[-1].append(offset)
                state = 0
            newdata.append((sentence, entities))
            offset = 0
            sentence = []
            entities = []

            continue

        items = [line[0], line[1:]]

        char = items[0].strip()
        tag = items[1].strip()
        #print char
        #print tag
        sentence.append(char)
        if tag.endswith("NOM"):
            tag = "O"
        if tag.startswith("B-"):
            if state:
                entities[-1].append(offset)
            entities.append(["entity",tag[2:], offset])
            state = 1
        elif tag == "O":
            if state:
                entities[-1].append(offset)
                state = 0
        offset += 1

    endata = []
    sennum = len(newdata)
    entitynum = sum([len(item[1]) for item in newdata])
    charnum  = sum(len(item[0]) for item in newdata)

    print "entity  number sum  :  "+str(entitynum)
    print "sentence number sum  :  "+str(sennum)
    print "char numbert  sum :  "+str(charnum)
    types = set()
    for item in newdata:
        for entity in item[1]:
            types.add(entity[1])

    print "entity types  : "+str(types)

    for item in newdata:
        entities = item[1]
        sentence = item[0]
        tags = ["O"]*len(sentence)
        for entity in entities:
            assert entity[2] < entity[3]
            if entity[3] - entity[2] == 1:
                tags[entity[2]] = "S-"+entity[1]
            else:
                tags[entity[2]] = "B-"+entity[1]
                tags[entity[3]- 1] = "E-"+entity[1]
                for index in xrange(entity[2]+1, entity[3]-1):
                    tags[index] = "I-"+entity[1]



        new_chars = []
        new_tags = []
        for i in range(len(sentence)):
            char = sentence[i].strip()
            if len(char) != 0:
                new_chars.append(re.sub("\d+\.\d+|\d+", "NUM", char))
                #new_words.append(word)
                new_tags.append(tags[i])

        endata.append([new_chars, new_tags])

    return  endata


def get_sighan_wdata(filename):

    data = io.open(filename, encoding = 'utf-8')

    sentence = ""
    entities = []
    offset = 0
    state = 0
    newdata = []
    for line in data:
        if line == "\n":
            if state:
                entities[-1].append(offset)
                state = 0
            newdata.append((sentence, entities))
            offset = 0
            sentence = ""
            entities = []

            continue

        items = [line[0], line[1:]]

        char = items[0].strip()
        tag = items[1].strip()
        #print char
        #print tag
        sentence += char
        if tag.startswith("B-"):
            if state:
                entities[-1].append(offset)
            entities.append(["entity",tag[2:], offset])
            state = 1
        elif tag == "O":
            if state:
                entities[-1].append(offset)
                state = 0
        offset += 1

    endata = []
    sennum = len(newdata)
    entitynum = sum([len(item[1]) for item in newdata])

    print "entity  number sum  :  "+str(entitynum)
    print "sentence number sum  :  "+str(sennum)
    errorseg = 0
    for item in newdata:
        entities = item[1]
        sentence = item[0]

        for entity in entities:
            assert len(entity) == 4

        entities.append([1,1,len(sentence)+1,len(sentence)+1])
        wordite = jieba.cut(sentence)
        #print " ".join(wordite).encode("utf-8")

        eindex = 0
        words = []
        tags = []
        loc = 0

        try:

            for word in wordite:
                words.append(word)
                n = 1
                if loc <  entities[eindex][2] and loc + len(words[-1]) <= entities[eindex][2]:
                    tags.append("NONE")

                elif loc <  entities[eindex][2]:
                    errorseg += 1
                    cword = words.pop(-1)
                    words.append(cword[0:entities[eindex][2] - loc])
                    n += 1
                    tags.append("NONE")
                    if  loc + len(cword) < entities[eindex][3]:
                        tags.append("B-"+entities[eindex][1])
                        words.append(cword[entities[eindex][2] - loc:len(cword)] )
                    elif loc + len(cword) == entities[eindex][3]:
                        tags.append("S-"+entities[eindex][1])
                        words.append(cword[entities[eindex][2] - loc:len(cword)] )
                        eindex += 1
                    else :

                        tags.append("S-"+entities[eindex][1])
                        tags.append("NONE")
                        n += 1
                        words.append(cword[entities[eindex][2] - loc: entities[eindex][3] - loc] )
                        words.append(cword[entities[eindex][3] - loc: len(cword) ] )

                        eindex +=1
                elif loc == entities[eindex][2] and loc + len(words[-1]) == entities[eindex][3]:
                    tags.append("S-"+entities[eindex][1])
                    eindex += 1

                elif loc ==  entities[eindex][2] and loc + len(words[-1]) < entities[eindex][3]:
                    tags.append("B-"+entities[eindex][1])

                elif loc ==  entities[eindex][2] and loc + len(words[-1]) > entities[eindex][3]:

                    cword = words.pop(-1)
                    words.append(cword[0:entities[eindex][3] - loc])
                    n += 1
                    words.append(cword[entities[eindex][3] - loc:len(cword)])
                    tags.append("S-"+entities[eindex][1])
                    eindex += 1
                    tags.append("NONE")
                    errorseg += 1

                elif loc >= entities[eindex][3]:# RIGHT
                    tags.append("NONE")

                elif loc + len(words[-1]) <  entities[eindex][3]:
                    tags.append("I-"+entities[eindex][1])


                elif loc + len(words[-1]) ==  entities[eindex][3]:
                    tags.append("E-"+entities[eindex][1])
                    eindex +=1
                else:
                    cword = words.pop(-1)
                    n += 1
                    words.append(cword[0:entities[eindex][3] - loc])
                    words.append(cword[entities[eindex][3] - loc:len(cword)])
                    tags.append("E-"+entities[eindex][1])
                    eindex += 1
                    tags.append("NONE")
                    errorseg += 1

                for k in range(1, n+1):
                    loc += len(words[0 - k])


        except Exception:
            for item in words:
                print item.encode('utf-8')

            print " target   entity" + "\n"
            for item in entities:
                print item[0].encode('utf-8')
            print entities[eindex][0].encode('utf-8')
            os._exit(1)



        new_words = []
        new_tags = []
        for i in range(len(words)):
            word = words[i].strip()
            if len(word) != 0:
                new_words.append(re.sub("\d+\.\d+|\d+", "NUM", word))
                #new_words.append(word)
                new_tags.append(tags[i])

        #print (" ".join(new_words)).encode("utf-8")
        #print (" ".join(new_words)).encode("utf-8")
        #test
        '''
        for word, tag in zip(new_words, new_tags):
            print word.encode('utf-8')+"  "+tag.encode('utf-8')
            print "........................................."
        '''
        endata.append((new_words,new_tags))

    print "word segementation error number sum  :" +str(errorseg)
    return endata


def getwData(filename):

    train = io.open(filename, encoding = 'utf-8')
    errorseg = []
    data = []
    error = 0
    iid = 0

    cha_num = 0
    entity_num = 0
    nums = []
    for line in train:
        iid += 1
        lpos = line.find("{{")

        entities = []
        while lpos >= 0:
            colon = line.find(":",lpos)
            tag = line[lpos+2: colon]
            rpos = line.find("}}", lpos)
            entity = line[colon + 1:rpos]
            entities.append((entity,tag,lpos,lpos+len(entity)))
            line = line[0:lpos] + entity + line[rpos+2:]
            lpos = line.find("{{")
            entity_num += 1

        new_entities = []
        for entity in entities:
            if entity[1] != "time":
                new_entities.append(entity)

        entities = new_entities
        nums.append(len(entities))
        cha_num += len(line)
        #  print line
        #  for item in entities:
        #     print item[0]
        # os._exit(1)
        entities.append((1,1,len(line)+1,len(line)+1))
        wordite = jieba.cut(line)
        #print " ".join(wordite).encode("utf-8")
        eindex = 0
        words = []
        tags = []
        loc = 0

        try:

            for word in wordite:
                words.append(word)
                n = 1
                if loc <  entities[eindex][2] and loc + len(words[-1]) <= entities[eindex][2]:
                    tags.append("NONE")

                elif loc <  entities[eindex][2]:

                    cword = words.pop(-1)
                    words.append(cword[0:entities[eindex][2] - loc])
                    n += 1
                    tags.append("NONE")
                    if  loc + len(cword) < entities[eindex][3]:
                        tags.append("B-"+entities[eindex][1])
                        words.append(cword[entities[eindex][2] - loc:len(cword)] )
                    elif loc + len(cword) == entities[eindex][3]:
                        tags.append("S-"+entities[eindex][1])
                        words.append(cword[entities[eindex][2] - loc:len(cword)] )
                        eindex += 1
                    else :

                        tags.append("S-"+entities[eindex][1])
                        tags.append("NONE")
                        n += 1
                        words.append(cword[entities[eindex][2] - loc: entities[eindex][3] - loc] )
                        words.append(cword[entities[eindex][3] - loc: len(cword) ] )

                        eindex +=1
                elif loc == entities[eindex][2] and loc + len(words[-1]) == entities[eindex][3]:
                    tags.append("S-"+entities[eindex][1])
                    eindex += 1

                elif loc ==  entities[eindex][2] and loc + len(words[-1]) < entities[eindex][3]:
                    tags.append("B-"+entities[eindex][1])

                elif loc ==  entities[eindex][2] and loc + len(words[-1]) > entities[eindex][3]:

                    cword = words.pop(-1)
                    words.append(cword[0:entities[eindex][3] - loc])
                    n += 1
                    words.append(cword[entities[eindex][3] - loc:len(cword)])
                    tags.append("S-"+entities[eindex][1])
                    eindex += 1
                    tags.append("NONE")

                elif loc >= entities[eindex][3]:# RIGHT
                    tags.append("NONE")

                elif loc + len(words[-1]) <  entities[eindex][3]:
                    tags.append("I-"+entities[eindex][1])


                elif loc + len(words[-1]) ==  entities[eindex][3]:
                    tags.append("E-"+entities[eindex][1])
                    eindex +=1
                else:
                    cword = words.pop(-1)
                    n += 1
                    words.append(cword[0:entities[eindex][3] - loc])
                    words.append(cword[entities[eindex][3] - loc:len(cword)])
                    tags.append("E-"+entities[eindex][1])
                    eindex += 1
                    tags.append("NONE")

                for k in range(1, n+1):
                    loc += len(words[0 - k])


        except Exception:
            for item in words:
                print item.encode('utf-8')

            print " target   entity" + "\n"
            for item in entities:
                print item[0].encode('utf-8')
            print entities[eindex][0].encode('utf-8')
            os._exit(1)



        new_words = []
        new_tags = []
        for i in range(len(words)):
            word = words[i].strip()
            if len(word) != 0:
                new_words.append(re.sub("\d+\.\d+|\d+", "NUM", word))
                #new_words.append(word)
                new_tags.append(tags[i])
                #print new_words[-1].encode("utf-8")+"  "+new_tags[-1].encode("utf-8")

        #print (" ".join(new_words)).encode("utf-8")
        #print (" ".join(new_words)).encode("utf-8")
        data.append((new_words,new_tags))

    print " development data entity number :  "+str(sum(nums[len(nums)/20*16:len(nums)/20*18]))
    print "word segementation error number :" + str(error)
    print "character  number :  " + str(cha_num)
    print "entity  number  :  " + str(entity_num)



    '''
    print "entity set......."
    for item in entities:
        print item[0]



    print "entity print over .............."

    '''

    return data

def tags2dic(ltags): #统计 lwords(lists of words) 构造词典
    dic = {}
    for tags in ltags:
        for tag in tags:
            dic.setdefault(tag,len(dic))
    return dic


def words2dic1(lwords,dic_size): #统计 lwords(lists of words) 构造词典

    return items2dic1(lwords, dic_size)
def chars2dic1(lchars, dic_size): #统计 lchars(lists of chars) build dic

    return items2dic1(lchars, dic_size)

def items2dic1(litems,dic_size): #统计 lwords(lists of items) 构造词典
    dic = {}
    for items in litems:
        for item in items:
            dic.setdefault(item,0)
            dic[item] +=1

    itemslist = dic.items()

    itemslist.sort(lambda x,y : cmp(y[1], x[1]))
    d = {}
    isize = min(len(itemslist), dic_size)
    for i in range(isize):
        d[itemslist[i][0]] = i+1

    return d


def words2dic2(filename, worddim):
    return items2dic2(filename, worddim)


def chars2dic2(filename, chardim):
    return items2dic2(filename, chardim)


def items2dic2(filename, itemdim):#word2vec tool  初始化词典
    d = {}
    f =io.open(filename, encoding = "utf-8")
    itemvecs = []
    itemvecs.append(np.random.randn(itemdim))
    dim = int(f.readline().split(" ")[1].strip())
    assert itemdim == dim
    i = 1
    for line in f:
        items = line.split(" ")
        d[items[0].strip()] = i
        itemvecs.append(np.asarray(map(lambda x: float(x),items[1:dim+1]), dtype = np.float32))
        i += 1
    return d, itemvecs

def raw2num2(rawdata, worddic,chardic, tagdic, window_size):

    traindata = []
    for item in rawdata:
        tagvec = []
        for tag in item[1]:
            tagvec.append(tagdic[tag])

        sen = item[0]
        for i in range(window_size):
            sen.insert(0,"start")
            sen.append("end")
        veclist = []#word list
        clist = [] #characters list each item contains indexs of all characters  in a word
        for word in sen:
            worddic.setdefault(word,0)
            veclist.append(worddic[word])
            chars = []
            for char in word:
                chardic.setdefault(char,0)
                chars.append(chardic[char])
            clist.append(chars)

        windex = []
        cindex = []
        for i in range(window_size, len(sen) - window_size):
            ivec = veclist[i-window_size:i+window_size+1]
            cvecs = clist[i - window_size:i+window_size+1]
            cvec = []
            for item in cvecs:
                cvec.extend(item)
            windex.append(ivec)
            cindex.append(cvec)


        traindata.append(((windex, cindex),tagvec))
    #windex consists of a list of items, each item is a word index vector corresponding to a point
    #cindex consists of a list of items, each item is a character index vector corresponding to a point

    return traindata

def raw2num1(rawdata, tokendic, tagdic, window_size, padding_id):
    # rawdata is consisted of  tuples of token seqeunces( word or characters)
    # and tag sequences([I,am,a,boy],[NONE, NONE,NONE,NONE])
    traindata = []
    for item in rawdata:
        tagvec = np.asarray([tagdic[tag] for tag in item[1]], dtype = np.int32)

        sen = item[0]
        tokenvec = []# token index list ,each token  could  be a word or a character
        for token in sen:
            tokendic.setdefault(token,0)
            tokenvec.append(tokendic[token])

        tokenvec = [padding_id for i in range(window_size)] + tokenvec + [padding_id for i in range(window_size)]

        windex = [ np.asarray(tokenvec[i-window_size:i+window_size+1], dtype = np.int32)  for i in range(window_size, len(sen) - window_size)]

        traindata.append(([windex],tagvec))
    #windex consists of a list of items, each item is a item  index vector corresponding to a point

    return traindata

def randomdata(data):

    n = len(data) - 1
    while n > 0:
        randindex = random.randint(0, n - 1)
        mid = data[n]
        data[n] = data[randindex]
        data[randindex] = mid
        n -= 1
    return data

def data2batch(data, bsize, pids):#data : ((em1,em2..), tag) padding_id :[ id1, id2..]
    start = 0
    groups = []
    maxlen = max(len(item[1]) for item in data)
    minlen = min(len(item[1]) for item in data)

    gap = (maxlen-minlen) / 10

    dic = {}
    for i in xrange(minlen, maxlen+1, gap):
        dic[i] = []

    for item in data:
        dic[(len(item[1]) - minlen)/gap*gap+minlen].append(item)


    def padgroup(group):
        maxlen = max(len(item[1]) for item in group)
        newgroup = []
        for instance in group:
            clen = len(instance[1])
            x = []
            for em,pid in zip(instance[0], pids):
                assert  len(em) == clen
                x.append(em + [ np.ones_like(em[0], dtype = np.int32)*pid for i in  range(maxlen - clen)])
            tags = np.pad(instance[1], (0,maxlen - clen),'constant',constant_values = (-1,-1))#tag for padding is -1

            newgroup.append((x, tags))
        return newgroup


    for key in dic:
        groups.extend([ padgroup(dic[key][start:start+bsize]) for start in xrange(0,len(dic[key])-bsize, bsize)])

    return groups


def tokenize(text):

    ite = jieba.cut(text)
    words = []
    for item in ite:
        word = item.strip()
        if len(word) != 0:
            words.append(re.sub("\d+\.\d+|\d+", "NUM", word))


    return words



if __name__ == "__main__":
    get_weibo_cdata(sys.argv[1])
