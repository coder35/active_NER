from collections import defaultdict

def computePR(tags, gold, predict):# gold and predict are a list of tag sequences
    tool = {}
    enname = {}
    for key in tags:
        if key != "NONE":
            entity = key[key.index("-")+1:]
            enname.setdefault(entity, 0)
            enname[entity] += 1

    for key in enname:
        tool[tags["S-"+key]] = [tags["S-"+key]]
        tool[tags["B-"+key]] = [tags["B-"+key], tags["I-"+key], tags["E-"+key],tags["S-"+key]]

    correct = 0.
    candidate = 0.
    extract = 0.

    dic = {}
    for key in tags:
        if key.startswith("S-"):
            dic[tags[key]] = [0.,0.,0.]# gold , predict, correct


    for i in range(len(gold)):

        golden = scantags(tool, gold[i])

        # gather all extracted entities from top n predicts
        preden = []
        for pre in predict[i]:
            if len(gold[i]) != len(pre):
                raise Exception
            items = scantags(tool, pre)
            for item in items:
                flag = True
                for exist in preden:
                    if item[0] == exist[0] and item[1] == exist[1] and item[2] == exist[2]:
                        flag = False
                        break
                if flag:
                    preden.append(item)

        correcten = common_en(golden, preden)
        correct += len(correcten)
        extract += len(preden)
        candidate += len(golden)


        for item in golden:
            dic[item[2]][0] += 1
        for item in preden:
            dic[item[2]][1] += 1
        for item in correcten:
            dic[item[2]][2] += 1
       

    pr_type = {}
    for key in dic:
        stat = dic[key]
        print stat
        pr_type[key] = [stat[2]/(stat[1] + 0.1), stat[2]/stat[0]]
    pr_type["overall"] = [correct / (extract+0.1), correct/ candidate] #precision and recall

    return pr_type

def tri(gold, predict1,predict2):# gold and predict are a list of tag sequences

    correct = 0.
    candidate = 0.
    extract = 0.
    common = 0.
    common_cor = 0.
    for i in range(len(gold)):

        golden = gold[i]

        # gather all extracted entities from top n predictsi of algorithm 1
        preden1 = predict1[i]
        # gather all extracted entities from top n predictsi of algorithm 2
        preden2 = predict2[i]

        common_pre = common_en(preden1, preden2)
        single = common_num(common_pre, golden)

        common += len(common_pre)
        common_cor += single
        candidate += len(golden)
        extract += len(preden1) + len(preden2) - len(common_pre)
        correct += common_num(golden, preden1) + common_num(golden, preden2) - single

    overall_p = correct/(extract+0.1)
    overall_r = correct/candidate
    agreement = common / extract
    agree_p = common_cor / common

    return [overall_p, overall_r, agreement, agree_p]  #precision and recall

def common_num(ens1,ens2):
    common = 0
    for e1 in ens1:
        for e2 in ens2:
            if e1[0] == e2[0] and e1[1] == e2[1] and e1[2] == e2[2]:
                common +=1
                break
    return common

def common_en(ens1, ens2):
    common = []
    for e1 in ens1:
        for e2 in ens2:
            if e1[0] == e2[0] and e1[1] == e2[1] and e1[2] == e2[2]:
                common.append(e1)
                break
    return common


def pres2en(tagdic,preseqs):
    #input a list of candidate predicate sequences

    preden = []
    for pre in preseqs:

        items = pre2en(tagdic, pre)
        for item in items:
            flag = True
            for exist in preden:
                if item[0]==exist[0] and item[1] == exist[1] and item[2] == exist[2]:
                    flag = False
                    break
            if flag:
                preden.append(item)
    return preden


def pre2en(tagdic,preseq):

    #input a candidate predicate sequence
    tool = {}
    enname = {}
    for key in tagdic:
        if key != "NONE":
            entity = key[key.index("-")+1:]
            enname.setdefault(entity, 0)
            enname[entity] += 1

    for key in enname:
        tool[tagdic["S-"+key]] = [tagdic["S-"+key]]
        tool[tagdic["B-"+key]] = [tagdic["B-"+key], tagdic["I-"+key], tagdic["E-"+key],tagdic["S-"+key]]


    return scantags(tool, preseq)

#eturn a list of entities, each is a tuple (start, end, type)
def scantags(tool, sen):
    entities = []
    i = 0
    while i < len(sen):
        if sen[i] in tool:
            en = tool[sen[i]]
            if len(en) == 1:
                entities.append((i,i+1,en[0]))
            else :
                k = i+1
                while k < len(sen) and sen[k] == en[1] : k = k+1
                if k < len(sen) and sen[k] == en[2]:
                    entities.append((i, k+1, en[3]))
                    i = k
                else : i = k - 1
        i += 1

    return entities

def extract_ens(tags, predict):
    tool = {}
    enname = set()
    for key in tags:
        if key != "NONE":
            entity = key[key.index("-")+1:]
            enname.add(entity)

    for key in enname:
        tool[tags["S-"+key]] = [tags["S-"+key]]
        tool[tags["B-"+key]] = [tags["B-"+key], tags["I-"+key], tags["E-"+key],tags["S-"+key]]

    extracted_ens = []
    for item in predict:
        assert len(item) == 1
        for ins in item:
            extracted_ens.append(scantags(tool, ins))

    result = []

    rev = dict(zip(tags.values(), tags.keys()))
    for item in extracted_ens:
        result.append([])
        for en in item:
            result[-1].append((en[0],en[1],rev[en[2]]))



    return result

