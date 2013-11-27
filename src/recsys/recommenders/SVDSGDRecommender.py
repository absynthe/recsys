import numpy as np
import scipy.sparse as sparse
import sys
import time
from test import cython_factorize_plain, cython_factorize_optimized
from recsys.base import BaseRecommender

class SVDSGDRecommender(BaseRecommender):
    def __init__(self,data, iterations=5000, factors=2, lr=0.001, reg= 0.02, with_preference=False):
        BaseRecommender.__init__(self, data, with_preference)
        self.p=None
        self.q=None
        self.average_rating = self.data.mean()
        #self.factorize_optimized(K = factors, steps = iterations, regularization = reg, learning_rate=lr)
        #self.factorize_plain(K = factors, steps = iterations, regularization = reg, learning_rate=lr)
        #self.p, self.q = cython_factorize_plain(self.data, factors, iterations, lr, reg)
        self.p, self.q = cython_factorize_optimized(self.data, factors, iterations, lr, reg)

    def factorize_optimized(self, K,steps=5000, learning_rate =0.001, regularization = 0.02, biased = False):
        #one default : 5000 steps and learning rate 0.0002
        # Koren uses 30 iterations, learning rate of 0.002 and regularization of 0.04
        """
        Factorizes the input matrix so as to minimize the regularized squared error,
        using the Singular Value Decomposition method
        @type R - sparse.csr_matrix
        @param R - the ratings matrix
        @type K - integer
        @param K -number of features
        @type steps - integer
        @param steps - number of steps
        @type learning_rate - float
        @param learning_rate - learning rate; Usually a small value. If too large may lead to
        oscillating around the minimum.
        @type regularization - float
        @param regularization - regularization constant;
        @rtype: np.vector
        @return: the user and item factor matrices
        """
        print "Computing factorizations..."
        print K
        #initialize factor matrices with random values
        N = self.data.shape[0] #no of users
        M = self.data.shape[1] #no of items
        #self.p = np.random.rand(N, K)
        #self.q = np.random.rand(M, K)

        self.p=np.empty([N,K])
        self.q=np.empty([M,K])
        self.p.fill(0.1)
        self.q.fill(0.1)
        self.q= self.q.T

        rowcols = np.array(self.data.nonzero())
        average_time = 0.0
        for step in xrange(steps):
            #SOMEWHAT OPTIMAL
            start_time = time.time()
            for u, i in rowcols.T:
                e= learning_rate * (self.data[u,i]-np.dot(self.p[u,:],self.q[:,i])) #calculate error for gradient
                #if biased:
                #     e -= self.average_rating
                p_temp = e * self.q[:,i] - learning_rate * regularization * self.p[u,:]
                #self.q[:,i]+= learning_rate * (e * self.p[u,:] - regularization * self.q[:,i])
                self.q[:,i]*=(1-learning_rate * regularization)
                self.q[:,i]+= e * self.p[u,:]
                self.p[u,:] += p_temp
            average_time +=time.time() - start_time
        sys.stdout.flush()
        print "One optimized step took on average" + str(average_time/steps), "seconds"


    def recommend(self,user_id, how_many):
        return

    def predict(self, user_id, item_id):
        return np.dot(self.p[user_id-1,:],self.q[:,item_id-1])

    def factorize_plain(self, K,steps=5000, learning_rate =0.001, regularization = 0.02, biased = False):
        """
        Factorizes the input matrix so as to minimize the regularized squared error,
        using the Singular Value Decomposition method
        @type R - sparse.csr_matrix
        @param R - the ratings matrix
        @type K - integer
        @param K -number of features
        @type steps - integer
        @param steps - number of steps
        @type learning_rate - float
        @param learning_rate - learning rate; Usually a small value. If too large may lead to
        oscillating around the minimum.
        @type regularization - float
        @param regularization - regularization constant;
        @rtype: np.vector
        @return: the user and item factor vectors
        """
        print "Computing factorizations..."
        print K
        N = self.data.shape[0] #no of users
        M = self.data.shape[1] #no of items

        #P = np.random.rand(N, K)
        #Q = np.random.rand(M, K)
        self.p=np.empty([N,K])
        self.q=np.empty([M,K])
        self.p.fill(0.1)
        self.q.fill(0.1)
        self.q= self.q.T
        rows,cols = self.data.nonzero()
        average_time = 0.0
        for step in xrange(steps):
            start_time = time.time()
            for u, i in zip(rows,cols):
                e_ui=self.data[u,i]-np.dot(self.p[u,:],self.q[:,i]) #calculate error for gradient
                for k in xrange(K):
                    # adjust P and Q based on error gradient
                    temp =self.p[u,k] + learning_rate * (e_ui * self.q[k,i] - regularization * self.p[u,k])
                    self.q[k,i]=self.q[k,i] + learning_rate * (e_ui * self.p[u,k] - regularization * self.q[k,i])
                    self.p[u,k]= temp
            average_time +=time.time() - start_time
        sys.stdout.flush()
        print "One plain step took on average" + str(average_time/steps), " seconds"