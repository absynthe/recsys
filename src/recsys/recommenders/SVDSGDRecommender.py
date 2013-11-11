import numpy as np
import scipy.sparse as sparse

def recommend(user_id, movie_id, p, q):
    nR = np.dot(p,q.T)
    return nR[user_id,movie_id]

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
    @return: the P and Q factor vectors
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