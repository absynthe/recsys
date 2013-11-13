import numpy as np
import scipy.sparse as sparse
from recsys.base import BaseRecommender

class SVDSGDRecommender(BaseRecommender):
    def __init__(self,model,with_preference=False):
        BaseRecommender.__init__(self, model, with_preference)
        self.p=None
        self.q=None
        self.average_rating = self.model.mean()
        self.factorize(K = 2)

    def factorize(self, K,steps=100, learning_rate =0.01, regularization = 0.02, biased = False):
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

        #initialize factor matrices with random values
        N = self.model.shape[0] #no of users
        M = self.model.shape[1] #no of items
        self.p = np.random.rand(N, K)
        self.q = np.random.rand(M, K)

        self.q = self.q #we transpose Q
        rows,cols = self.model.nonzero()

        for step in xrange(steps):
            for u, i in zip(rows,cols):
                e=self.model-np.dot(self.p,self.q.T) #calculate error for gradient
                if biased:
                     e -= self.average_rating
                #for k in xrange(K):
                    # adjust P and Q based on error gradient
                    #temp =self.p[u,k] + learning_rate * (2 * e[u,i] * self.q[k,i] - regularization * self.p[u,k])
                    #self.q[k,i]+= learning_rate * (2 * e[u,i] * self.p[u,k] - regularization * self.q[k,i])
                    #self.p[u,k]= temp
                p_temp = learning_rate * (2 * e[u,i] * self.q[i,:] - regularization * self.p[u,:])
                self.q[i,:]+= learning_rate * (2 * e[u,i] * self.p[u,:] - regularization * self.q[i,:])
                self.p[u,:] += p_temp
                #compute square error
                #error=0
                #for u, i in zip(rows,cols):
                #    error = error + pow(self.model[u,i] - np.dot(self.p[u,:],self.q[:,i]),2)
                #    if regularization >0:
                #        error = error + (regularization/2) * (np.sum(np.square(self.p[u,:]))+np.sum(np.square(self.q[:,i])))
                #dot = np.dot(self.p,self.q.T)
                #error =  np.sum(np.power(self.model[rows,cols] - dot[rows,cols],2))
                #if regularization > 0:
                #    error *= regularization/2
                #if error < 0.001:
                #    break
        #params are now learnt

    def opt_factorize(self, K=2,steps=5000, learning_rate =0.0002, regularization = 0.02):
        N = self.model.shape[0] #no of users
        M = self.model.shape[1] #no of items
        self.p = np.random.rand(N, K)
        temp = np.zeros((N, K))
        self.q = np.random.rand(M, K)
        error = sparse.csr_matrix(self.model)

        #self.q = self.q.T we transpose Q
        rows,cols = self.model.nonzero()
        urows = np.unique(rows)
        ucols = np.unique(cols)

        for step in xrange(steps):
            #calculate an initial error matrix
            dot = np.dot(self.p,self.q.T)
            error[rows,cols] = self.model[rows,cols]-dot[rows,cols]
            temp = self.p[urows,:] + learning_rate * (2 * error[rows,cols] * self.q[ucols,:])
            self.q[ucols,:]+= learning_rate * (2*error[rows,cols]*self.p[urows,:])
            self.p = temp
            #calculate square error
            error = np.sum(pow(self.model[rows,cols] - np.dot(self.p[rows,:],self.q[cols,:].T),2))
            if error < 0.001:
                break
        #params are now learnt
        return np.dot(self.p, self.q.T)


    def recommend(self,user_id, how_many):
        return

    def predict(self, user_id, item_id):
        return np.dot(self.p[user_id-1,:],self.q[item_id-1,:].T)
        #return nR[user_id,movie_id]

    def mf_sgd(self,K=2,steps=5000, learning_rate =0.0002, regularization = 0.02): #0.02
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
        R = self.model
        N = R.shape[0] #no of users
        M = R.shape[1] #no of items

        P = np.random.rand(N, K)
        Q = np.random.rand(M, K)

        Q = Q.T #we transpose Q
        rows,cols = R.nonzero()

        for step in xrange(steps):
            for u, i in zip(rows,cols):
                e_ui=R[u,i]-np.dot(P[u,:],Q[:,i]) #calculate error for gradient
                for k in xrange(K):
                    # adjust P and Q based on error gradient
                    temp =P[u][k] + learning_rate * (2 * e_ui * Q[k][i] - regularization * P[u][k])
                    Q[k][i]=Q[k][i] + learning_rate * (2 * e_ui * P[u][k] - regularization * Q[k][i])
                    P[u][k]= temp
                #compute square error
                e=0
                for u, i in zip(rows,cols):
                    e = e + pow(R[u,i] - np.dot(P[u,:],Q[:,i]),2)
                    if regularization >0:
                        e = e + (regularization/2) * (np.sum(np.square(P[u,:]))+np.sum(np.square(Q[:,i])))
                if e < 0.001:
                    break
        return np.dot(P, Q)