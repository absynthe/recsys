import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
from recsys.data.base import load_movielens_ratings100k, generate,eval_movielens_test100k
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
    start_time = time.time()
    data = load_movielens_ratings100k()
    sys.stdout.flush()
    print "Model building took " + str(time.time() - start_time), "seconds"
    start_time = time.time()
    rec = SVDSGDRecommender(data, 0.002, 0.04, 2, 30)
    print "Factorization took " + str(time.time() - start_time), "seconds"
    start_time = time.time()
    eval_movielens_test100k(rec)
    print "Evaluation took " + str(time.time() - start_time), "seconds"
