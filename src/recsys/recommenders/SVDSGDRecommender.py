import numpy as np
import scipy.sparse as sparse
import sys
import time
from test import cython_factorize_plain, cython_factorize_optimized
from recsys.base import BaseRecommender

class SVDSGDRecommender(BaseRecommender):
    def __init__(self,data,
                 iterations=5000, factors=2,
                 learning_rate=0.001, regularization= 0.02,
                 with_bias = False, bias_learning_rate = 0.001, bias_regularization = 0.02,
                 with_feedback=False):
        """
        @type data - sparse.csr_matrix
        @param data - the ratings matrix in CSR format
        @type iterations - integer
        @param iterations - number of steps
        @type features- integer
        @param factors -number of features
        @type learning_rate - float
        @param learning_rate - learning rate; Usually a small value. If too large may lead to
        oscillation around the minimum.
        @type regularization - float
        @param regularization - regularization constant;
        @type with_bias - boolean
        @param with_bias - specifies whether to take user-item bias into account
        @type bias_learning_rate - float
        @param bias_learning_rate - lerning rate or bias
        @type bias_regularization - float
        @param bias_regularization - regularization constant for bias
        @type with_feedback - boolean
        @param with_feedback - specifies whether to take user-item bias into account

        #one default : 5000 steps and learning rate 0.0002
        # Koren uses 30 iterations, learning rate of 0.002 and regularization of 0.04
        """
        BaseRecommender.__init__(self, data)
        self.with_feedback = with_feedback
        self.with_bias= with_bias
        if (self.with_bias):
            self.global_average = self.data.mean()
            self.user_bias = np.zeros(self.no_users)
            self.item_bias = np.zeros(self.no_items)
        if (self.with_feedback):
            #self.feedback_data =  sparse.lil_matrix(self.data, dtype=np.float64)
            #self.feedback_data[self.feedback_data.nonzero()]=1
            #self.feedback_data.tocsr()
            self.y = None
        #self.factorize_optimized(iterations, factors, learning_rate, regularization, self.with_bias, bias_learning_rate, bias_regularization, self.with_feedback)
        #self.factorize_plain(iterations, factors, learning_rate, regularization)
        #self.p, self.q = cython_factorize_plain(self.data, factors, iterations, lr, reg)
        self.p, self.q = cython_factorize_optimized(self.data, factors, iterations, learning_rate, regularization)

    def recommend(self,user_id, how_many):
        return

    def predict(self, user_id, item_id):
        return np.dot(self.p[user_id-1,:],self.q[:,item_id-1])

    def factorize_optimized(self,
                            steps=5000, K = 2,
                            learning_rate =0.001, regularization = 0.02,
                            biased = False, bias_learning_rate = 0.001, bias_regularization=0.02,
                            feedback = False):
        """
        Factorizes the input matrix so as to minimize the regularized squared error,
        using the Stochastic Gradient Descent method. With bias and feedback, represents SVD ++.
        Optimized version.
        """
        print "Computing factorizations for " + str(K) + "factors"

        #initialize factor matrices with random values or fixed values
        #self.p = np.random.rand(N, K)
        #self.q = np.random.rand(M, K)
        self.p=np.empty([self.no_users,K])
        self.q=np.empty([self.no_items,K])
        self.p.fill(0.1)
        self.q.fill(0.1)
        if self.with_feedback:
            self.y = np.empty([self.no_items,K])
            self.y.fill(0.1)

        rowcols = np.array(self.data.nonzero())
        average_time = 0.0
        for step in xrange(steps):
            start_time = time.time()
            for u, i in rowcols.T:
                if self.with_feedback:
                    # calculate y.SumOfRows(items_rated_by_user[u]);
                    py_sum = np.empty(K)
                    item_indices = self.data[u].nonzero()[1]
                    denominator = math.sqrt(item_indices.size)
                    for it in item_indices:
                        py_sum += self.y[it,:]
                    #normalize
                    py_sum /= denominator
                    #add to p
                    py_sum += self.p[u,:]
                    #calculate error for gradient
                    e = (self.data[u,i]-np.dot(py_sum,self.q.T[:,i]))
                else:
                    #directly calculate error for gradient
                    e= (self.data[u,i]-np.dot(self.p[u,:],self.q.T[:,i]))
                #take bias into account
                if self.with_bias:
                     e -= ( self.global_average + self.user_bias[u] + self.item_bias[i] )
                     item_bias[i] += bias_learning_rate * (e - bias_regularization * item_bias[i])
                     user_bias[u] += bias_learning_rate * (e - bias_regularization * user_bias[u])
                p_temp = learning_rate * (e * self.q[i,:] - regularization * self.p[u,:])
                #adjust p, q and y factors
                if self.with_feedback:
                    #adjust y first
                    for it in item_indices:
                        self.y[it,:] *= (1 - learning_rate * regularization)
                        self.y[it,:] += learning_rate * e / denominator * q[i,:]
                    # then q
                    self.q[i,:]*=(1-learning_rate * regularization)
                    self.q[i,:]+= learning_rate * e * py_sum
                else:
                    #self.q[i,:]+= learning_rate * (e * self.p[u,:] - regularization * self.q[i,:])
                    self.q[i,:]*=(1-learning_rate * regularization)
                    self.q[i,:]+= learning_rate * e * self.p[u,:]
                self.p[u,:] += p_temp
            average_time +=time.time() - start_time
        sys.stdout.flush()
        print "One optimized step took on average" + str(average_time/steps), "seconds"
        self.q = self.q.T

    def factorize_plain(self,
                        steps=5000, K = 2,
                        learning_rate =0.001, regularization = 0.02):
        """
        Factorizes the input matrix so as to minimize the regularized squared error,
        using the Stochastic Gradient Descent method. Unoptimal version.
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
        """
        print "Computing factorizations for " + str(K) + "factors"


        #self.p = np.random.rand(N, K)
        #self.q = np.random.rand(M, K)
        self.p=np.empty([self.no_users,K])
        self.q=np.empty([self.no_items,K])
        self.p.fill(0.1)
        self.q.fill(0.1)
        self.q=self.q.T

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