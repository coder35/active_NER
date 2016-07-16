import matplotlib.pyplot as plt
import sys
import re
def getscores(filename):
    f = file(filename)
    epoch = []
    fscore = []
    for line in f:
        if not line.startswith("epoch"):
            continue
        m = re.match('epoch.*(\d).*(0\.\d+).*(0\.\d+).*(0\.\d+).*(0\.\d+)', line)
        epoch.append(int(m.group(1)))
        fscore.append(float(m.group(5)))
        return (epoch, fscore)



tsfile = sys.argv[1]
dsfile = sys.argv[2]

train = getscores(tsfile)
dev = getscores(dsfile)





