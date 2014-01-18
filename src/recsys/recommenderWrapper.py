import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
from recsys.data.base import load_movielens_ratings100k, generate,eval_movielens_test100k, cross_validate_movielens_test100k_iterations, cross_validate_movielens_test100k
from recsys.recommenders.SVDSGDRecommender import SVDSGDRecommender

ratings_matrix_cache = None
user_id_cache = None
p_cache = None
q_cache = None
chosen_model = None
MOVIES = 17770
USERS = 480189
PATH = '/Users/ana/Documents/Netflix Whole Dataset/training_set/mv_'

def convergence_lr():
    #plot convergence for learning rate of basic SGD on the training set
    plots = []
    for lr in [0.0001,0.001,0.01]:
        plots.append(cross_validate_movielens_test100k_iterations(25, 25, 1000, lr))
    print plots

def error_rate_lr():
    #plot error_rate depending on learning_rate on training and validation set
    train_plots = []
    validation_plots = []
    for lr in np.arange(0.001,0.01,0.001):#np.arange(0.001,0.01,0.001):
        print lr
        train, validation = cross_validate_movielens_test100k(300, 2, lr, 0)
        train_plots.append(train)
        validation_plots.append(validation)
    print train_plots
    print validation_plots

def error_rate_without_regularization():
    #plot error_rate depending on features for optimal learning_rate without regularization
    train_plots = []
    validation_plots = []
    for features in [2,3,4] + range(5,30,5):
        train, validation = cross_validate_movielens_test100k(300, features, 0.001, 0)
        train_plots.append(train)
        validation_plots.append(validation)
    print train_plots
    print validation_plots

def regularization_factors():
    #plot regularization effect depending on factors
    train_plots = []
    validation_plots = []
    for reg in [0.1]:
        t = []
        v = []
        for features in (range(2,20,2) + range(20, 140, 20)):#(range(2,20,2) + range(20, 140, 20)):
            train, validation = cross_validate_movielens_test100k(300, features, 0.001, reg)
            t.append(train)
            v.append(validation)
        train_plots.append(t)
        validation_plots.append(v)
    print train_plots
    print validation_plots

def compute_netflix():

    data, mapping = load_netflix_r(j+1, False)
    #train recommender
    rec = SVDSGDRecommender(data, steps, factors, lr, reg, False, 0, 0, False)
    print rec.rmse
    # estimate on probe data
    base_dir =
    test_ratings = np.loadtxt(base_dir + 'u' + str(id) + '.test', delimiter='\t', usecols=(0, 1, 2), dtype=int)
    total = 0
    n = 0
    for user_id, item_id, rating in test_ratings:
        estimate = rec.predict(user_id,item_id)
        total += math.pow(np.float64(rating)-estimate,2)
        n+=1
    rmse = math.sqrt(total/n)
    print "Rmse for fold " + str(id) + "is: " + str(rmse)
    return rmse
    # estimate on qualifying data ? CAN'T because it was only available during the contest



if __name__ == "__main__":
    #convergence_lr()
    #error_rate_lr() # found 0.006
    regularization_factors()
    #error_rate_without_regularization()

    #cross-validation for regularized SGD
    #for lr, reg in [(0.01,0.01),(0.001,0.01),(0.01,0.001),(0.001,0.001),(0.01,0.1),(0.001,0.1)]:

    #cross-validation for SVD

    #cross-validation for SVD++

    #cross_validate_movielens_test100k_factors()
    #cross_validate_movielens_test100k_reg()
    #cross_validate_movielens_test100k_bias()
    #start_time = time.time()
    #data = load_movielens_ratings100k()
    #data = generate()
    #print "Data building took " + str(time.time() - start_time), "seconds"
    #start_time = time.time()
    #data, iterations=5000, factors=2, lr=0.001, reg= 0.02, with_preference=False
    # for time testing :     rec = SVDSGDRecommender(data, 10, 200, 0.001, 0.02, False, 0.001, 0.02, False)
    #rec = SVDSGDRecommender(data, 5000, 4, 0.0002, 0.02, False, 0.0002, 0.02, False)
    #print rec.data
    #print "Factorization took " + str(time.time() - start_time), "seconds"
    #print np.dot(rec.p,rec.q)
    #start_time = time.time()
    #eval_movielens_test100k(rec)
    #print "Evaluation took " + str(time.time() - start_time), "seconds"



