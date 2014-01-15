import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
from recsys.data.base import load_movielens_ratings100k, generate,eval_movielens_test100k, cross_validate_movielens_test100k_iterations, cross_validate_movielens_test100k_factors,cross_validate_movielens_test100k_reg,cross_validate_movielens_test100k_bias
from recsys.recommenders.SVDSGDRecommender import SVDSGDRecommender

ratings_matrix_cache = None
user_id_cache = None
p_cache = None
q_cache = None
chosen_model = None
MOVIES = 17770
USERS = 480189
PATH = '/Users/ana/Documents/Netflix Whole Dataset/training_set/mv_'

if __name__ == "__main__":
    #cross-validation for basic SGD
    #for lr in [0.0001,0.001,0.01]:
    #    cross_validate_movielens_test100k_iterations(25, 25, 1000, lr)

    #cross-validation for regularized SGD
    #for lr, reg in [(0.01,0.01),(0.001,0.01),(0.01,0.001),(0.001,0.001),(0.01,0.1),(0.001,0.1)]:

    #cross-validation for SVD

    #cross-validation for SVD++

    cross_validate_movielens_test100k_factors()
    #cross_validate_movielens_test100k_reg()
    #cross_validate_movielens_test100k_bias()
    #start_time = time.time()
    #data = load_movielens_ratings100k()
    #data = generate()
    #print "Data building took " + str(time.time() - start_time), "seconds"
    #start_time = time.time()
    #data, iterations=5000, factors=2, lr=0.001, reg= 0.02, with_preference=False
    # for time testing :     rec = SVDSGDRecommender(data, 10, 200, 0.001, 0.02, False, 0.001, 0.02, False)
    #rec = SVDSGDRecommender(data, 2, 2, 0.001, 0, True, 0.001, 0.02, False)
    #print rec.data
    #print "Factorization took " + str(time.time() - start_time), "seconds"
    #print np.dot(rec.p,rec.q)
    #start_time = time.time()
    #eval_movielens_test100k(rec)
    #print "Evaluation took " + str(time.time() - start_time), "seconds"
