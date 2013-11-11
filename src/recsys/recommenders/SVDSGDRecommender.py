import numpy as np
import scipy.sparse as sparse
from recsys.base import BaseRecommender

class SVDSGDRecommender(BaseRecommender):
    def __init__(self,model,with_preference=False):
        BaseRecommender.__init__(self, model, with_preference)
        self.p=None
        self.q=None
        self.factorize(K = 2)

    def factorize(self, K,steps=5000, alpha =0.0002, beta = 0.02):
        """
        Factorizes the input matrix so as to minimize the regularized squared error,
        using the Singular Value Decomposition method
        @type R - sparse.csr_matrix
        @param R - the ratings matrix
        @type K - integer
        @param K -number of features
        @type steps - integer
        @param steps - number of steps
        @type alpha - float
        @param alpha - learning rate; Usually a small value. If too large may lead to
        oscillating around the minimum.
        @type beta - float
        @param beta - regularization constant;
        @rtype: np.vector
        @return: the user and item factor matrices
        """
        print "Computing factorizations..."

        #initialize factor matrices with random values
        N = self.model.shape[0] #no of users
        M = self.model.shape[1] #no of items
        self.p = np.random.rand(N, K)
        self.q = np.random.rand(M, K)

        self.q = self.q.T #we transpose Q
        rows,cols = self.model.nonzero()

        for step in xrange(steps):
            for u, i in zip(rows,cols):
                e_ui=self.model[u,i]-np.dot(self.p[u,:],self.q[:,i]) #calculate error for gradient
                for k in xrange(K):
                    # adjust P and Q based on error gradient
                    temp =self.p[u][k] + alpha * (2 * e_ui * self.q[k][i] - beta * self.p[u][k])
                    self.q[k][i]=self.q[k][i] + alpha * (2 * e_ui * self.p[u][k] - beta * self.q[k][i])
                    self.p[u][k]= temp
                #compute square error
                e=0
                for u, i in zip(rows,cols):
                    e = e + pow(self.model[u,i] - np.dot(self.p[u,:],self.q[:,i]),2)
                    if beta >0:
                        e = e + (beta/2) * (np.sum(np.square(self.p[u,:]))+np.sum(np.square(self.q[:,i])))
                if e < 0.001:
                    break
        #params are now learnt

    def recommend(self,user_id, how_many):
        return

    def predict(self, user_id, item_id):
        return np.dot(self.p[user_id-1,:],self.q[item_id-1,:].T)
        #return nR[user_id,movie_id]

def mf_sgd(R, K,steps=5000, alpha =0.0002, beta = 0.02): #0.02
    """
    Factorizes the input matrix so as to minimize the regularized squared error,
    using the Singular Value Decomposition method
    @type R - sparse.csr_matrix
    @param R - the ratings matrix
    @type K - integer
    @param K -number of features
    @type steps - integer
    @param steps - number of steps
    @type alpha - float
    @param alpha - learning rate; Usually a small value. If too large may lead to
    oscillating around the minimum.
    @type beta - float
    @param beta - regularization constant;
    @rtype: np.vector
    @return: the user and item factor vectors
    """
    print "Computing factorizations..."

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
                temp =P[u][k] + alpha * (2 * e_ui * Q[k][i] - beta * P[u][k])
                Q[k][i]=Q[k][i] + alpha * (2 * e_ui * P[u][k] - beta * Q[k][i])
                P[u][k]= temp
            #compute square error
            e=0
            for u, i in zip(rows,cols):
                e = e + pow(R[u,i] - np.dot(P[u,:],Q[:,i]),2)
                if beta >0:
                    e = e + (beta/2) * (np.sum(np.square(P[u,:]))+np.sum(np.square(Q[:,i])))
            if e < 0.001:
                break
    return P, Q.T