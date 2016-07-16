import test
import theano
import theano.tensor as T
import numpy as np
def haofunc(y,x):

    return x+[1,-1]*y

init = T.vector('s')
ve = T.vector('v')
a = np.arange(2)
result, up = theano.scan(fn = haofunc, outputs_info = np.zeros_like(a,dtype = np.float), sequences = [ve])

f = theano.function([ve], result)

print f([1,2,3,4,5])

