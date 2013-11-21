import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
cimport numpy as np

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

def factorize(data, int K,int steps=5000, np.float64_t learning_rate =0.001, np.float64_t regularization = 0.02):
    print "Computing factorizations..."

    assert data.dtype == DTYPE
    #initialize factor matrices with random values
    cdef int N = data.shape[0] #no of users
    cdef int M = data.shape[1] #no of items  
    cdef np.ndarray p = np.empty([N,K], dtype=DTYPE)
    cdef np.ndarray p_temp = np.empty([1,K], dtype=DTYPE)
    cdef np.ndarray q = np.empty([M,K], dtype=DTYPE)
    cdef np.float64_t e 
    cdef np.ndarray rowcol = np.array(data.nonzero(),dtype=int)
    cdef int step
    cdef int i
    cdef int u

    
    #p.fill(0.1)
    #q.fill(0.1)
    p = np.random.rand(N, K)
    q = np.random.rand(M, K)
    q= q.T
    rows,cols = data.nonzero()
    
    average_time = 0.0
    for step in xrange(steps):
        #SOMEWHAT OPTIMAL
        start_time = time.time()
        for u, i in rowcol.T:
            e=learning_rate * ( data[u,i]-np.dot(p[u,:],q[:,i])) #calculate error for gradient
            p_temp = ( e * q[:,i] - learning_rate * regularization * p[u,:])
            q[:,i]*=(1-learning_rate * regularization)
            q[:,i]+= e * p[u,:]
            p[u,:] += p_temp
        average_time +=time.time() - start_time
    print "One step took on average" + str(average_time/steps), "seconds"
    return p,q