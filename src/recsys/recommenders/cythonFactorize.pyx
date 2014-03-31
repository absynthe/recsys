#!python
#cython: profile=True
#cython: boundscheck=False
#cython: infer_types=True
#cython: wraparound=False
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
cimport numpy as np

from cpython cimport bool
import cython

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

def factorize_plain(data, int K,int steps=5000, np.float64_t learning_rate =0.001, np.float64_t regularization = 0.02):
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

def factorize_optimized(data, int K,int max_steps=5000,
    np.float64_t learning_rate =0.001, np.float64_t regularization = 0.02,
    bool early_stop = False, np.float64_t min_improvement = 0.0001,
    randomNoise = 0.05): #0.05 good for Movielens

    print "Computing factorizations..."

    #predefine all variables for efficient memory allocation
    assert data.dtype == DTYPE
    cdef unsigned int N = data.shape[0] #no of users
    cdef unsigned int M = data.shape[1] #no of items
    cdef np.ndarray[DTYPE_t,ndim=2] p = np.empty([N,K], dtype=DTYPE)
    cdef np.float64_t p_temp, estimated_rating
    cdef np.ndarray[DTYPE_t,ndim=2] q = np.empty([M,K], dtype=DTYPE)
    cdef np.float64_t e, oldRMSE, newRMSE
    cdef np.ndarray[long,ndim=2] rowcol = np.array(data.nonzero(),dtype=long)
    cdef np.ndarray[DTYPE_t,ndim=1] values = data.data
    cdef unsigned int step
    cdef unsigned int i
    cdef unsigned int u
    cdef unsigned int j, x
    cdef unsigned int dim = data.size
    cdef np.float64_t global_average = 0.0
    cdef np.float64_t initVal = 0.0

    #initialize factor matrices with random preseeded values
    #np.random.seed(1)
    #p = np.random.rand(N, K)
    #np.random.seed(2)
    #q = np.random.rand(M, K)

    ##initialize factor matrices with an uniform distribution centered around sqrt(mean divided by number of factors)
    #compute mean
    for x in xrange(dim):
        global_average += values[x]
    global_average /= dim

    print "Global average is:" + str(global_average)
    # init values
    initVal=np.sqrt(global_average/K)
    print "Initial values is" + str(initVal)
    np.random.seed(1)
    p = np.random.uniform(-0.05, 0.05, (N,K)) + initVal
    np.random.seed(2)
    q = np.random.uniform(-0.05, 0.05, (M,K)) + initVal

    #print p,q

    q= q.T

    if early_stop:
        # Compute initial RMSE
        oldRMSE = rmse(values,rowcol,p,q,dim,K)
        print "Initial training RMSE is : " + str(oldRMSE)

    #initialize time for iterations benchmark
    average_time = 0.0

    #actual training, all factors are trained at the same time
    for step in xrange(max_steps):
        start_time = time.time()
        for x in xrange(dim):
            u = rowcol[0,x]
            i = rowcol[1,x]
            estimated_rating = 1.0
            for j in xrange(K):#calculate error for gradient
                estimated_rating += p[u,j] * q[j,i]
                #clip
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
        if early_stop:
            #calculate new RMSE and compare with old RMSE
            newRMSE = rmse(values,rowcol,p,q,dim,K)
            if oldRMSE-newRMSE < min_improvement:
                print "Early stopping. Stable RMSE is:" + str(newRMSE) +" Number of iterations is: " + str(step+1)
                break
            oldRMSE = newRMSE

    print step
    if early_stop and step >= max_steps-1:
        print "Maximum number of iterations reached. RMSE is: " + str(newRMSE)

    if max_steps > 0:
        print "One step took on average" + str(average_time/(step+1)) + "seconds"

    if not early_stop or max_steps == 0:
        newRMSE = rmse(values,rowcol,p,q,dim,K)
        print "RMSE is: " + str(newRMSE)

    return p,q, newRMSE

def svd(data, int K,int max_steps=5000,
    np.float64_t learning_rate =0.001, np.float64_t regularization = 0.02,
    np.float64_t bias_learning_rate =0.001, np.float64_t bias_regularization = 0.02,
    early_stop = False, min_improvement = 0.0001):

    print "Computing factorizations with bias..."

    assert data.dtype == DTYPE

    cdef unsigned int N = data.shape[0] #no of users
    cdef unsigned int M = data.shape[1] #no of items
    cdef np.ndarray[long,ndim=2] rowcol = np.array(data.nonzero(),dtype=long)
    cdef np.ndarray[DTYPE_t,ndim=1] values = np.array(data.data,dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=2] p = np.empty([N,K], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] q = np.empty([M,K], dtype=DTYPE)
    np.random.seed(3)
    cdef np.ndarray[DTYPE_t,ndim=1] user_bias = np.random.rand(N)
    np.random.seed(4)
    cdef np.ndarray[DTYPE_t,ndim=1] item_bias = np.random.rand(M)
    cdef np.float64_t p_temp, estimated_rating
    cdef np.float64_t global_average = 0.0
    cdef np.float64_t total = 0.0
    cdef np.float64_t e

    cdef unsigned int step, j, x #iterators
    cdef unsigned int i, u #indices
    cdef unsigned int dim = data.size

    #compute mean
    for x in xrange(dim):
        global_average += values[x]
    global_average /= dim

    print "The mean is " + str(global_average)

    #correct gloabl_average because ratings between 1 and 5

    #initialize factor matrices with random values
    #np.random.seed(1)
    #p = np.random.rand(N, K)
    #np.random.seed(2)
    #q = np.random.rand(M, K)

    # init values
    initVal=np.sqrt(global_average/K)
    #print "Initial values is" + str(initVal)
    np.random.seed(1)
    p = np.random.uniform(-0.05, 0.05, (N,K)) + initVal
    #p = np.random.normal(0,0.1,(N,K))
    np.random.seed(2)
    #q = np.random.normal(0,0.1,(M,K))
    q = np.random.uniform(-0.05, 0.05, (M,K)) + initVal

    q= q.T

    if early_stop:
        # Compute initial RMSE
        oldRMSE = rmse_bias(values,rowcol,p,q,dim,K, global_average, user_bias, item_bias)
        print "Initial training RMSE is :" + str(oldRMSE)

    average_time = 0.0
    for step in xrange(max_steps):
        start_time = time.time()
        for x in xrange(dim):
            u = rowcol[0,x]
            i = rowcol[1,x]
            estimated_rating = global_average + user_bias[u] + item_bias[i]
            for j in xrange(K):#calculate error for gradient
                estimated_rating += p[u,j] * q[j,i]
                if estimated_rating < 1.0 :
                    estimated_rating = 1.0
                elif estimated_rating > 5.0:
                    estimated_rating = 5.0

            e = values[x]-estimated_rating


            #adjust biases
            item_bias[i] += bias_learning_rate * (e - bias_regularization * item_bias[i])
            user_bias[u] += bias_learning_rate * (e - bias_regularization * user_bias[u])

            #adjust factors
            for j in xrange(K):
                p_temp = learning_rate * ( e * q[j,i] - regularization * p[u,j])
                q[j,i] *= (1.0-learning_rate * regularization)
                q[j,i] += e * learning_rate * p[u,j]
                p[u,j] += p_temp
        #learning_rate = 0.9 * learning_rate
        #bias_learning_rate = 0.9 * bias_learning_rate
        average_time +=time.time() - start_time
        if early_stop:
            #calculate new RMSE and compare with old RMSE
            newRMSE = rmse_bias(values,rowcol,p,q,dim,K,global_average, user_bias, item_bias)
            if oldRMSE-newRMSE < min_improvement:
                print "Early stopping. Stable RMSE is:" + str(newRMSE) +" Number of iterations is:" + str(step+1)
                break
            oldRMSE = newRMSE

    if early_stop and step >= max_steps-1:
        print "Maximum number of iterations reached. RMSE is: " + str(newRMSE)

    if max_steps > 0:
        print "One step took on average " + str(average_time/(step+1)) + "seconds"

    if not early_stop or max_steps == 0:
        newRMSE = rmse_bias(values,rowcol,p,q,dim,K, global_average, user_bias, item_bias)

    return p,q, user_bias,item_bias, newRMSE, global_average


@cython.boundscheck(False)
def svd_plus_plus(
    int N,
    int M,
    int dim,
    np.ndarray[long, ndim=2] rowcol,
    np.ndarray[DTYPE_t, ndim=1] values,
    np.ndarray[long, ndim = 1] item_indices,
    np.ndarray[long, ndim = 1] user_pointers,
    int K,
    int max_steps=5000,
    np.float64_t learning_rate = 0.001,
    np.float64_t regularization = 0.02,
    np.float64_t bias_learning_rate =0.001,
    np.float64_t bias_regularization = 0.02,
    bool early_stop = False,
    np.float64_t min_improvement = 0.0001):

    print "SVD ++ : Computing factorizations with bias and feedback for" + str(K) + " factors"

    #assert data.dtype == DTYPE

    #cdef unsigned int N = data.shape[0] #no of users
    #cdef unsigned int M = data.shape[1] #no of items

    cdef np.ndarray[DTYPE_t,ndim=2] p = np.empty([N,K], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] q = np.empty([M,K], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] y = np.empty([M,K], dtype=DTYPE)

    np.random.seed(3)
    cdef np.ndarray[DTYPE_t,ndim=1] user_bias = np.random.rand(N)

    np.random.seed(4)
    cdef np.ndarray[DTYPE_t,ndim=1] item_bias = np.random.rand(M)

    cdef np.ndarray[DTYPE_t,ndim=1] py_sum = np.empty(K, dtype=DTYPE)

    #cdef np.ndarray[long, ndim = 1] temp_indices = np.empty(M, dtype=long)


    cdef np.float64_t estimated_rating
    cdef np.float64_t global_average = 0.0
    cdef np.float64_t total = 0.0
    cdef np.float64_t denominator = 0.0
    cdef np.float64_t e
    cdef np.float64_t val
    cdef np.float64_t start_time
    cdef np.float64_t  average_time = 0.0


    cdef long step, j, it, x #iterators
    cdef long i, u, last #indices
    cdef long ti, idx
    #cdef unsigned int dim = data.size

    # cache a dictionary of user-rated_items mappings

    users = set(rowcol[0,:])
    rated_items = []

    cdef np.ndarray[long, ndim=1] items, temp_indices
    for user in users:
        items = item_indices[user_pointers[user]:user_pointers[user+1]] if (user < user_pointers.size-1) else item_indices[user_pointers[user]:]
        rated_items.append(items)

    #compute mean
    for x in xrange(dim):
        global_average += values[x]
    global_average /= dim

    print "The mean is " + str(global_average)

    #initialize factor matrices with random values
    #np.random.seed(1)
    #p = np.random.rand(N, K)
    #np.random.seed(2)
    #q = np.random.rand(M, K)

    # initialize values with normal / uniform distibution
    initVal=np.sqrt(global_average/K)
    #print "Initial value is" + str(initVal)
    np.random.seed(1)
    p = np.random.uniform(-0.05, 0.05, (N,K)) + initVal #p = np.random.normal(0,0.1,(N,K))
    np.random.seed(2)
    q = np.random.uniform(-0.05, 0.05, (M,K)) + initVal #q = np.random.normal(0,0.1,(M,K))
    np.random.seed(5)
    y = np.random.uniform(-0.05, 0.05, (M,K)) + initVal #y = np.random.normal(0,0.1,(M,K))

    q= q.T

    if early_stop:
        # Compute initial RMSE
        oldRMSE = rmse_feedback(values,rowcol,item_indices, user_pointers, p,q,y,dim,K, global_average, user_bias, item_bias)
        print "Initial training RMSE is :" + str(oldRMSE)

    last = rowcol[0,0]
    print 'last = ', last
    temp_indices = rated_items[last]
    denominator = np.sqrt(temp_indices.size)

    for step in xrange(max_steps):

        print "Step:",step
        start_time = time.time()

        for x in xrange(dim):
            u = rowcol[0,x]
            i = rowcol[1,x]

            #add biases
            estimated_rating = global_average + user_bias[u] + item_bias[i]

            # calculate y.SumOfRows(items_rated_by_user[u]);
            py_sum = np.zeros(K, dtype=DTYPE)
            #item_indices = np.array(data.getrow(u).indices, dtype = long)

            if u != last:
                temp_indices = rated_items[u]#item_indices[user_pointers[u]:user_pointers[u+1]] if (u < user_pointers.size-1) else item_indices[user_pointers[u]:]
                denominator = np.sqrt(temp_indices.size)
            #print user_pointers[u],user_pointers[u+1],denominator
            #print u, temp_indices

            for j in xrange(K):
                for ti in xrange(temp_indices.size):
                    py_sum[j] += y[temp_indices[ti],j]

                #normalize
                py_sum[j] /= denominator
                #add p to it
                py_sum[j] += p[u,j]

                #calculate error for gradient
                #e = (self.data[u,i]-np.dot(py_sum,self.q.T[:,i]))

                estimated_rating += py_sum[j] * q[j,i]
                if estimated_rating < 1.0 :
                    estimated_rating = 1.0
                elif estimated_rating > 5.0:
                    estimated_rating = 5.0

            e = values[x] - estimated_rating

            #adjust biases
            item_bias[i] += bias_learning_rate * (e - bias_regularization * item_bias[i])
            user_bias[u] += bias_learning_rate * (e - bias_regularization * user_bias[u])

            #adjust p, q, y factors

            #adjust y first
            for j in xrange(K):
                for ti in xrange(temp_indices.size):
                    idx = temp_indices[ti]
                    y[idx,j] *= (1 - learning_rate * regularization)
                    y[idx,j] += learning_rate * e / denominator * q[j,i]

            # then q and p

                #p_temp = learning_rate * ( e * q[j,i] - regularization * p[u,j])
                q[j,i] *= (1.0-learning_rate * regularization)
                q[j,i] += e * learning_rate * py_sum[j]
                p[u,j] += learning_rate * ( e * q[j,i] - regularization * p[u,j])

        average_time += time.time() - start_time

        if early_stop:
            #calculate new RMSE and compare with old RMSE
            newRMSE = rmse_feedback(values,rowcol,item_indices, user_pointers, p,q,y,dim,K, global_average, user_bias, item_bias)
            if oldRMSE-newRMSE < min_improvement:
                print "Early stopping. Stable RMSE is:" + str(newRMSE) +" Number of iterations is:" + str(step+1)
                break
            oldRMSE = newRMSE

    if early_stop and step >= max_steps-1:
        print "Maximum number of iterations reached. RMSE is: " + str(newRMSE)

    if max_steps > 0:
        print "One step took on average " + str(average_time/(step+1)) + "seconds"

    if not early_stop or max_steps == 0:
        newRMSE = rmse_feedback(values,rowcol,item_indices, user_pointers, p,q,y,dim,K, global_average, user_bias, item_bias)

    print "newRMSE = ", newRMSE

    return p, q , y, user_bias,item_bias, newRMSE, global_average


####### Helper functions

@cython.boundscheck(False)
cdef np.float64_t svdpp_inner_loop(
    np.ndarray[long, ndim=1] temp_indices,
    np.ndarray[DTYPE_t,ndim=1] py_sum,
    np.ndarray[DTYPE_t,ndim=2] y,
    np.float64_t denominator,
    long j,
    long u,
    long i,
    np.float64_t estimated_rating,
    np.ndarray[DTYPE_t,ndim=2] p,
    np.ndarray[DTYPE_t,ndim=2] q):

    cdef long ti
    for ti in xrange(temp_indices.size):
        py_sum[j] += y[temp_indices[ti],j]

    #normalize
    py_sum[j] /= denominator
    #add p to it
    py_sum[j] += p[u,j]

    #calculate error for gradient
    #e = (self.data[u,i]-np.dot(py_sum,self.q.T[:,i]))

    estimated_rating += py_sum[j] * q[j,i]
    if estimated_rating < 1.0 :
        estimated_rating = 1.0
    elif estimated_rating > 5.0:
        estimated_rating = 5.0

    return estimated_rating

cdef svdpp_inner_loop2(
    np.ndarray[long, ndim = 1] temp_indices,
    np.ndarray[DTYPE_t,ndim=2] p,
    np.ndarray[DTYPE_t,ndim=2] q,
    np.ndarray[DTYPE_t,ndim=2] y,
    long i,
    long j,
    long u,
    np.float64_t learning_rate,
    np.float64_t regularization,
    np.float64_t e,
    np.float64_t denominator,
    np.ndarray[DTYPE_t,ndim=1] py_sum):

    cdef long ti, idx
    for ti in xrange(temp_indices.size):
        idx = temp_indices[ti]
        y[idx,j] *= (1 - learning_rate * regularization)
        y[idx,j] += learning_rate * e / denominator * q[j,i]

# then q and p

    #p_temp = learning_rate * ( e * q[j,i] - regularization * p[u,j])
    q[j,i] *= (1.0-learning_rate * regularization)
    q[j,i] += e * learning_rate * py_sum[j]
    p[u,j] += learning_rate * ( e * q[j,i] - regularization * p[u,j])


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

def clamped_predict_bias(np.ndarray[DTYPE_t,ndim=1] p_row,np.ndarray[DTYPE_t,ndim=1] q_row,
    np.float64_t min_val,np.float64_t max_val,
    np.float64_t global_average,np.float64_t user_bias,np.float64_t item_bias):
    #as recommended by Funk
    cdef np.float64_t estimated_rating = global_average + item_bias + user_bias
    cdef unsigned int k
    for k in range(p_row.size):
        estimated_rating += p_row[k] * q_row[k]
        if estimated_rating < min_val :
            estimated_rating = min_val
        elif estimated_rating > max_val:
            estimated_rating = max_val
    return estimated_rating

def clamped_predict_feedback(np.ndarray[DTYPE_t,ndim=1] p_row,np.ndarray[DTYPE_t,ndim=1] q_row,
    np.ndarray[DTYPE_t,ndim=2] y, np.ndarray[int,ndim=1] item_indices,
    np.float64_t min_val,np.float64_t max_val,
    np.float64_t global_average,np.float64_t user_bias,np.float64_t item_bias):

    cdef np.float64_t estimated_rating = global_average + item_bias + user_bias
    cdef np.float64_t denominator = 0.0
    cdef unsigned int k,it
    cdef np.ndarray[DTYPE_t,ndim=1] py_sum = np.zeros(p_row.size, dtype=DTYPE)

    # calculate y.SumOfRows(items_rated_by_user[u]);
    denominator = np.sqrt(item_indices.size)
    for it in item_indices:
        for k in xrange(p_row.size):
            py_sum[k] += y[it,k]

    for k in range(p_row.size):
        estimated_rating += py_sum[k] * q_row[k]
        if estimated_rating < min_val :
            estimated_rating = min_val
        elif estimated_rating > max_val:
            estimated_rating = max_val
    return estimated_rating

cdef np.float64_t rmse(np.ndarray[DTYPE_t,ndim=1] values, np.ndarray[long,ndim=2] rowcol,
    np.ndarray[DTYPE_t,ndim=2] p, np.ndarray[DTYPE_t,ndim=2] q,
    unsigned int dim, int K):
    # calculate RMSE for the recommender and return it

    cdef np.float64_t estimated_rating
    cdef np.float64_t total = 0.0
    cdef unsigned int x, u, i, j

    for x in xrange(dim):
        u = rowcol[0,x]
        i = rowcol[1,x]
        estimated_rating = 1.0
        for j in xrange(K):#calculate error for gradient
            estimated_rating += p[u,j] * q[j,i]
            #clip
            if estimated_rating < 1.0 :
                estimated_rating = 1.0
            elif estimated_rating > 5.0:
                estimated_rating = 5.0

        total += (values[x]-estimated_rating)**2
    return np.sqrt(total/np.float64(dim))

cdef np.float64_t rmse_bias(np.ndarray[DTYPE_t,ndim=1] values, np.ndarray[long,ndim=2] rowcol,
    np.ndarray[DTYPE_t,ndim=2] p, np.ndarray[DTYPE_t,ndim=2] q,
    unsigned int dim, int K,
    np.float64_t global_average,
    np.ndarray[DTYPE_t,ndim=1] user_bias, np.ndarray[DTYPE_t,ndim=1] item_bias):
    # calculate RMSE for the recommender and return it

    cdef np.float64_t estimated_rating
    cdef np.float64_t total = 0.0
    cdef unsigned int x, u, i, j

    for x in xrange(dim):
        u = rowcol[0,x]
        i = rowcol[1,x]
        estimated_rating = global_average + user_bias[u] + item_bias[i]
        for j in xrange(K):#calculate error for gradient
            estimated_rating += p[u,j] * q[j,i]
            #clip
            if estimated_rating < 1.0 :
                estimated_rating = 1.0
            elif estimated_rating > 5.0:
                estimated_rating = 5.0
        total += (values[x]-estimated_rating)**2
    return np.sqrt(total/np.float64(dim))

cdef np.float64_t rmse_feedback(
    np.ndarray[DTYPE_t,ndim=1] values,
    np.ndarray[long,ndim=2] rowcol,
    np.ndarray[long, ndim = 1] item_indices,
    np.ndarray[long, ndim = 1] user_pointers,
    np.ndarray[DTYPE_t,ndim=2] p,
    np.ndarray[DTYPE_t,ndim=2] q,
    np.ndarray[DTYPE_t,ndim=2] y,
    unsigned int dim,
    int K,
    np.float64_t global_average,
    np.ndarray[DTYPE_t,ndim=1] user_bias,
    np.ndarray[DTYPE_t,ndim=1] item_bias):
    # calculate RMSE for the recommender and return it

    cdef np.float64_t estimated_rating
    cdef np.ndarray[DTYPE_t,ndim=1] py_sum = np.empty(K)
    cdef np.float64_t total = 0.0
    cdef np.float64_t denominator = 0.0
    cdef unsigned int x, u, i, j,it, idx, w

    for x in xrange(dim):
        u = rowcol[0,x]
        i = rowcol[1,x]

        estimated_rating = global_average + user_bias[u] + item_bias[i]

        # calculate y.SumOfRows(items_rated_by_user[u]);
        py_sum = np.zeros(K)

        if u < user_pointers.size - 1:
            temp_indices = item_indices[user_pointers[u]:user_pointers[u+1]]
        else:
            temp_indices = item_indices[user_pointers[u]:]


        denominator = np.sqrt(temp_indices.size)
        for it in xrange(temp_indices.size):
            idx = temp_indices[it]
            for w in xrange(K):
                py_sum[w] += y[idx,w]

        for j in xrange(K):
            #normalize
            py_sum[j] /= denominator
            #add p to it
            py_sum[j] += p[u,j]

            #calculate error for gradient
            #e = (self.data[u,i]-np.dot(py_sum,self.q.T[:,i]))

            estimated_rating += py_sum[j] * q[j,i]
            if estimated_rating < 1.0 :
                estimated_rating = 1.0
            elif estimated_rating > 5.0:
                estimated_rating = 5.0
        total += (values[x]-estimated_rating)**2

    return np.sqrt(total / np.float64(dim))
