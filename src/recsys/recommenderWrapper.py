import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
from recsys.data.base import load_movielens_r100k
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
    data = load_movielens_r100k()
    print "Model building took " + str(time.time() - start_time), "seconds"
    start_time = time.time()
    rec = SVDSGDRecommender(data)
    print "Factorization took " + str(time.time() - start_time), "seconds"
    start_time = time.time()
    print rec.predict(1, 17)
    print "Prediction took"  + str(time.time() - start_time), "seconds"
#    while True:
#        if ratings_matrix_cache is None or user_id_cache is None:
#            start_time = time.time()
#            ratings_matrix_cache, user_id_cache = recommender.generate()
#            #ratings_matrix_cache, user_id_cache = recommender.generate_ratings_matrix(MOVIES,USERS,PATH)
#            print time.time() - start_time, "seconds"
#            print ratings_matrix_cache
#        print "What do you want to do? \n reload \n recommend \n model"
#        command = sys.stdin.readline()
#        command = command.rstrip('\n')
#        if command == "reload":
#            print "reloading..."
#            reload(recommender)
#            os.system(['clear','cls'][os.name == 'nt'])
#        elif command == "recommend":
#            if p_cache is not None and q_cache is not None:
#                recommender.recommend(user_id,movie_id,p_cache,q_cache,user_id_cache)
#            else:
#                print "No model detected! Please build model first!"
#        elif command == "model":
#            print "Choose the desired model: \n svd \n svdreg"
#            choice = sys.stdin.readline()
#            choice = choice.rstrip('\n')
#            if choice !=chosen_model:
#                if choice == "svd":
#                    print ratings_matrix_cache
#                    p_cache, q_cache = recommender.mf_svd_noreg(ratings_matrix_cache, 2)
#                elif choice =="svdreg":
#                    p_cache, q_cache = recommender.mf_svd_reg(ratings_matrix_cache, 2)
#                print "Model built"
#                print p_cache, q_cache
#                chosen_model=choice
#                sys.stdin.readline()
#            else:
#                print "Model already initialized. Keeping data."
#                sys.stdin.readline()