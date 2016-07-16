import numpy as np
import random
import time
import re
def k_means(items,k):
    #input items is matrix, each row is a data point
    #output clusters is a dictionary , value at index k (cluster id) is a list of vector index in  matrix items  eg. clusters[2] = [1,2,4]


    new_spoints = choose_sp(items,k)
    old_spoints = np.zeros_like(new_spoints)
    print "start k-means clustering ............"
    epoch = 0
    dis = 1
    while dis  > 1e-1:
        clusters = {}
        for i in xrange(k):
    	    clusters[i] = []
        for index in xrange(len(items)):
            vec = items[index]
            mincluster = np.argmin([np.sum((vec-point)**2) for point in new_spoints])
            clusters[mincluster].append(index)

        old_spoints = new_spoints
        new_spoints = []
        for i in xrange(k):
            if len(clusters[i]) == 0:
                new_spoints.append(np.zeros_like(old_spoints[0]))
                continue
            new_spoints.append(np.sum(items[clusters[i]], 0)/len(clusters[i]))

        epoch += 1
        dis = sum([ np.sum((new_spoints[i]-old_spoints[i])**2) for i in xrange(len(new_spoints))])
        print time.strftime('%Y-%m-%d %H:%M:%S')+"   epoch   :     "+str(epoch)+ "   difference   :   "+str(dis)
        if epoch == 50:
            break
    new_clusters = {}
    i = 0
    for index in clusters:
	if len(clusters[index]) > 0:
	    new_clusters[i] = clusters[index]
	    i += 1
    clusters = new_clusters
    return clusters

def choose_sp(items,k):
    minvec = np.min(items, 0)
    maxvec = np.max(items, 0)

    k_midpoints = [np.arange(minvec[i]+(maxvec[i]-minvec[i])/(2*k), maxvec[i],(maxvec[i] - minvec[i])/k) for i in xrange(len(minvec))]

    spoints = []
    while len(spoints) != k:
        spoints.append(np.array([  k_midpoints[i][random.randint(0,len(k_midpoints[i])-1)] for i in xrange(len(k_midpoints))], dtype = np.float32))

    return spoints

def assign_label(data,neg_label,embeddings,label_para = -1, clus_type = "num",context = 0):
    
    subnum = 4
    ma = []
    length = len(data[0][1])
    for ins in data:
        
        assert ins[1][length / 2] == neg_label
        embedding = sum(embeddings[ins[0][i]] for i in xrange(length/2-context, length/2+context+1)) / (1+2*context)
        assert len(embedding) == len(embeddings[0])
        ma.append(embedding)

    ma = np.array(ma, dtype = np.float32)
    print "clustered data  point number ................"+str(len(ma))
    if clus_type == "num":
	print "clustering according to cluster number........"+str(label_para) 
        label_num = label_para # clustering number 
        clusters = k_means(ma, label_num)
    elif clus_type == "size":
	print "clustering according to cluster size........ "+str(label_para)
        maxrate = label_para # max cluster size
        maxsize = len(ma)*maxrate
        clusters = k_means(ma, subnum)
	mid = {}
	for midindex in clusters:
	    mid[midindex] = len(clusters[midindex])
	print mid
        while True:
            flag = True
            s = len(clusters)
            for key in xrange(s):
                if len(clusters[key]) > maxsize:
                    subma = ma[clusters[key]]
                    subclusters = k_means(subma, subnum)
		    mid = {}
		    for midindex in subclusters:
			mid[midindex] = len(subclusters[midindex])
		    print mid
                    for subkey in subclusters:
                        original = []
                        for index in subclusters[subkey]:
                            original.append(clusters[key][index])
                        subclusters[subkey] = original

                    clusters[key] = subclusters[0]

                    start = len(clusters)
                    for i in xrange(1,len(subclusters)):
                        clusters[start+i-1] = subclusters[i]
                    flag = False

            if flag:
                break


    for label in clusters:
        for index in clusters[label]:
            data[index][1][length/2]  = label + neg_label
  
    return len(clusters)

def resetdata(data, neg_label):
    length = len(data[0][1])
    for ins in data:
        if ins[1][length/2] >= neg_label:
            ins[1][length/2] = neg_label


def randomdata(data):
    for i in xrange(len(data) - 1):
        randindex = random.randint(i+1, len(data) - 1)
        data[i], data[randindex] = data[randindex], data[i]



def get_bestepoch(scoref):
    f = open(scoref)
    bestf = 0.
    for line in f:
        r = re.match("\D+(\d)\D+(\d\.\d+)\D+(\d\.\d+)\D+(\d\.\d+)", line)
        epoch = int(r.group(1))
        f = float(r.group(4))
        if f > bestf:
            beste = epoch
            bestf = f

    return beste



'''
a = np.arange(1000).reshape((100,10))
clusters = k_means(a,10)

for key in clusters:
    print str(key)+"  :  "+str(len(clusters[key]))

'''


