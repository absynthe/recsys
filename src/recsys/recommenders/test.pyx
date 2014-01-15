import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
cimport numpy as np

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

def cython_factorize_plain(data, int K,int steps=5000, np.float64_t learning_rate =0.001, np.float64_t regularization = 0.02):
    print "Computing factorizations..."
    print K
    assert data.dtype == DTYPE
    #initialize factor matrices with random values
    cdef unsigned int N = data.shape[0] #no of users
    cdef unsigned int M = data.shape[1] #no of items  
    cdef np.ndarray[DTYPE_t,ndim=2] p = np.empty([N,K], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] p_temp = np.empty(K, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] q = np.empty([M,K], dtype=DTYPE)
    cdef np.float64_t e 
    cdef np.ndarray[long,ndim=2] rowcol = np.array(data.nonzero(),dtype=long)
    cdef unsigned int step
    cdef unsigned int i
    cdef unsigned int u

    p.fill(0.1)
    q.fill(0.1)
    #p = np.random.rand(N, K)
    #q = np.random.rand(M, K)
    q= q.T
    
    average_time = 0.0
    for step in xrange(steps):
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
    
def cython_factorize_optimized(data, int K,int steps=5000, np.float64_t learning_rate =0.001, np.float64_t regularization = 0.02):
    print "Computing factorizations..."
    print K
    assert data.dtype == DTYPE
    #initialize factor matrices with random values
    cdef unsigned int N = data.shape[0] #no of users
    cdef unsigned int M = data.shape[1] #no of items  
    cdef np.ndarray[DTYPE_t,ndim=2] p = np.empty([N,K], dtype=DTYPE)
    cdef np.float64_t p_temp, estimated_rating
    cdef np.ndarray[DTYPE_t,ndim=2] q = np.empty([M,K], dtype=DTYPE)
    cdef np.float64_t e 
    cdef np.ndarray[long,ndim=2] rowcol = np.array(data.nonzero(),dtype=long)
    cdef np.ndarray[DTYPE_t,ndim=1] values = data.data
    cdef unsigned int step
    cdef unsigned int i
    cdef unsigned int u
    cdef unsigned int j
    cdef unsigned int dim = data.size

    
    p.fill(0.1)
    q.fill(0.1)
    #p = np.random.rand(N, K)
    #q = np.random.rand(M, K)
    q= q.T
    
    average_time = 0.0
    for step in xrange(steps):
        start_time = time.time()
        for x in xrange(dim):
            u = rowcol[0,x]
            i = rowcol[1,x]
            estimated_rating = 1.0
            for j in xrange(K):#calculate error for gradient
                estimated_rating += p[u,j] * q[j,i]
                #clamp
                if estimated_rating < 1.0 :
                    estimated_rating = 1.0
                elif estimated_rating > 5.0:
                    estimated_rating = 5.0
            e=learning_rate * (values[x]-estimated_rating) 
            for j in xrange(K):
                p_temp = ( e * q[j,i] - learning_rate * regularization * p[u,j])
                q[j,i]*=(1.0-learning_rate * regularization)
                q[j,i]+= e * p[u,j]
                p[u,j] += p_temp
        average_time +=time.time() - start_time
    print "One step took on average" + str(average_time/steps), "seconds"
    return p,q
    
def cython_factorize_optimized_biased(data, np.float64_t global_average,
                                      int K,int steps=5000, 
                                      np.float64_t learning_rate =0.001, np.float64_t regularization = 0.02,
                                      np.float64_t bias_learning_rate =0.001, np.float64_t bias_regularization = 0.02):
    print "Computing factorizations..."
    print K
    assert data.dtype == DTYPE
    #initialize factor matrices with random values
    cdef unsigned int N = data.shape[0] #no of users
    cdef unsigned int M = data.shape[1] #no of items  
    cdef np.ndarray[long,ndim=2] rowcol = np.array(data.nonzero(),dtype=long)
    cdef np.ndarray[DTYPE_t,ndim=1] values = data.data
    
    cdef np.ndarray[DTYPE_t,ndim=2] p = np.empty([N,K], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] q = np.empty([M,K], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] user_bias = np.zeros(N)
    cdef np.ndarray[DTYPE_t,ndim=1] item_bias = np.zeros(M)
    cdef np.float64_t p_temp, estimated_rating
    cdef np.float64_t e 

    cdef unsigned int step
    cdef unsigned int i
    cdef unsigned int u
    cdef unsigned int j
    cdef unsigned int dim = data.size

    
    p.fill(0.1)
    q.fill(0.1)
    #p = np.random.rand(N, K)
    #q = np.random.rand(M, K)
    q= q.T
    
    average_time = 0.0
    for step in xrange(steps):
        start_time = time.time()
        for x in xrange(dim):
            u = rowcol[0,x]
            i = rowcol[1,x]
            estimated_rating = 0.0
            for j in xrange(K):#calculate error for gradient
                estimated_rating += p[u,j] * q[j,i]
            estimated_rating += global_average + user_bias[u] + item_bias[i]
            e= values[x]-estimated_rating
            #adjust biases 
            item_bias[i] += bias_learning_rate * (e - bias_regularization * item_bias[i])
            user_bias[u] += bias_learning_rate * (e - bias_regularization * user_bias[u])
            for j in xrange(K):
                p_temp = learning_rate * ( e * q[j,i] - regularization * p[u,j])
                q[j,i]*=(1.0-learning_rate * regularization)
                q[j,i]+= e * learning_rate * p[u,j]
                p[u,j] += p_temp
        average_time +=time.time() - start_time
    print "One step took on average" + str(average_time/steps), "seconds"
    return p,q,user_bias,item_bias
    
def clamped_predict(np.ndarray[DTYPE_t,ndim=1] p_row,np.ndarray[DTYPE_t,ndim=1] q_row,np.float64_t min_val,np.float64_t max_val):
    #as recommended by Funk
    cdef np.float64_t estimated_rating = min_val
    cdef unsigned int k
    for k in range(p_row.size):
        estimated_rating += p_row[k] * q_row[k] 
        if estimated_rating < min_val :
            estimated_rating = min_val
        elif estimated_rating > max_val:
            estimated_rating = max_val 
    return estimated_rating
    