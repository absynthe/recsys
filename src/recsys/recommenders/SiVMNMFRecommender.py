import numpy as np
import scipy.sparse as sparse
import sys
import time
from recsys.recommenders.pymf import SIVM
from recsys.base import BaseRecommender

class SiVMNMFRecommender(BaseRecommender):
    def __init__(self, data,
                 iterations=5000, factors=4, clipping = True):
        """
        @type data - sparse.csr_matrix
        @param data - the ratings matrix in CSR format
        @type iterations - integer
        @param iterations - number of steps
        @type factors - integer
        @param factors - number of features
        """

        BaseRecommender.__init__(self, data)

        self.clipping = clipping
        self.model = SIVM(self.data, num_bases=factors)
        total_time = 0.0
        for i in range(10):
            start_time = time.time()
            self.model.factorize(show_progress=True, compute_w=True, compute_h=True,
                      compute_err=True, niter=iterations)
            total_time +=time.time() - start_time
        print "Factorization took " + str(total_time/10.0) + "seconds"
        self.w = self.model.W.todense()
        self.h = self.model.H


    def predict(self, user_id, item_id):
        prediction = np.dot(self.w[user_id-1], self.h[:,item_id-1])
        if self.clipping:
            if prediction >= 5.0:
                return 5.0
            elif prediction < 1.0:
                return 1.0
        return prediction[0,0]