# -*- coding: utf-8 -*-
import math
from os.path import dirname
from os.path import join
import sys, os
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import time
from recsys.data.base import load_movielens_ratings100k, generate,eval_movielens_test100k, cross_validate_movielens_test100k_iterations, cross_validate_movielens_test100k, load_netflix_r_pretty, cross_validate_movielens_test100k_sivm
from recsys.recommenders.SVDSGDRecommender import SVDSGDRecommender
from recsys.recommenders.SiVMNMFRecommender import SiVMNMFRecommender

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
    for lr in [0.001]:#[0.0001,0.001,0.01]:
        print "Computing for learning rate" + str(lr)
        plots.append(cross_validate_movielens_test100k_iterations(25, 25, 500, lr))
    print plots

def error_rate_lr():
    #plot error_rate depending on learning_rate on training and validation set
    train_plots = []
    validation_plots = []
    for lr in np.arange(0.001,0.01,0.001):#np.arange(0.001,0.01,0.001):
        print "Computing for learning rate" + str(lr)
        train, validation = cross_validate_movielens_test100k(300, 2, lr, 0, False, lr, 0)
        train_plots.append(train)
        validation_plots.append(validation)
    print train_plots
    print validation_plots

def error_rate_without_regularization():
    #plot error_rate depending on features for optimal learning_rate without regularization
    train_plots = []
    validation_plots = []
    for features in [2,10,20,50,100,200]:#[2,3,4] + range(5,30,5):
        train, validation = cross_validate_movielens_test100k(300, features, 0.002, 0, False, 0, 0)
        train_plots.append(train)
        validation_plots.append(validation)
    print train_plots
    print validation_plots

def regularization_factors():
    #plot regularization effect depending on factors
    train_plots = []
    validation_plots = []
    for reg in [0.1]:#[0.0015, 0.002, 0.003, 0.03, 0.06, 0.1]:
        t = []
        v = []
        for features in [2,10,20,50,100,200]:#[10,20,50,100,200]:#(range(2,22,2) + range(20, 150, 20) + [200]):#(range(2,20,2) + range(20, 140, 20)):
            train, validation = cross_validate_movielens_test100k(300, features, 0.002, reg, False, 0, 0)
            t.append(train)
            v.append(validation)
        train_plots.append(t)
        validation_plots.append(v)
    print train_plots
    print validation_plots

def bias_test():
    #plot regularization effect depending on factors
    train_plots = []
    validation_plots = []
    for reg in [0.1]:
        t = []
        v = []
        for features in [200]:#[2,10,20,50,100,200]:#(range(2,22,2) + range(20, 150, 20) + [200]):#(range(2,20,2) + range(20, 140, 20)):
            train, validation = cross_validate_movielens_test100k(300, features, 0.002, reg, True, 0.002, 0.001, False)
            t.append(train)
            v.append(validation)
        train_plots.append(t)
        validation_plots.append(v)
    print train_plots
    print validation_plots

def feedback_test():
    #plot regularization effect depending on factors
    train_plots = []
    validation_plots = []
    for reg in [0.011]:
        t = []
        v = []
        for features in [10]:#[2,10,20,50,100,200]:#(range(2,22,2) + range(20, 150, 20) + [200]):#(range(2,20,2) + range(20, 140, 20)):
            train, validation = cross_validate_movielens_test100k(10, features, 0.002, reg, True, 0.001, 0.0011, True)
            #cythonFactorize.svd_plus_plus(\ no_users, no_items, no_ratings,\ rowcol, values, item_indices, user_pointers,\ 10, 10, 0.002, 0.011, 0.001, 0.0011, True)
            t.append(train)
            v.append(validation)
        train_plots.append(t)
        validation_plots.append(v)
    print train_plots
    print validation_plots

def compute_netflix(data, steps = 200, factors = 10, lr = 0.001, reg = 0.011, bias = False, lrb = 0.001, regb=0.011):
    #train recommender
    print "Training the recommender SGD with " + str(steps) +" iterations and " + str(factors) + " factors"
    rec = SVDSGDRecommender(data, steps, factors, lr, reg, bias, lrb, regb, False)
    #rec = SVDSGDRecommender(data, 30, 10, 0.005, 0.02, True, 0.005, 0.02, False)
    # estimate on probe data
    print "Estimating RMSE on probe set."
    test_ratings = np.loadtxt('/Users/ana/workspace/recsys/src/recsys/data/raw/pretty-probe.txt', delimiter=' ', usecols=(0, 1, 2), dtype=int)
    total = 0.0
    n = 0
    for user_id, item_id, rating in test_ratings:
        estimate = rec.predict(user_id,item_id)
        total += math.pow(np.float64(rating)-estimate,2)
        n+=1
    rmse = math.sqrt(total/n)
    print "RMSE for training data is:" + str(rec.rmse)
    print "RMSE for probe data is: " + str(rmse)
    return rmse
    # estimate on qualifying data ? CAN'T because it was only available during the contest

if __name__ == "__main__" :
    print "Ready"
    #convergence_lr()
    #error_rate_lr()
    #regularization_factors()
    #error_rate_without_regularization()
    feedback_test()

    #data = generate()
    #rec = SVDSGDRecommender(data, 10, 10, 0.002, 0.1, True, 0.002, 0.001, True)

    #print "Loading data"
    #data = load_netflix_r_pretty()
    #compute_netflix(data, 200,10,0.001,0.02,True,)
    #compute_netflix(data, 40,10,0.005,0.02,True,0.005,0.02)
#    0.005, λ = 0.02
    #compute_netflix(data, 200,64)
    #compute_netflix(data, 200,128)
    #compute_netflix(data, 30)

    #data = generate()
    #rec = SVDSGDRecommender(data, 5000, 3, 0.0002, 0.02, True, 0.0002, 0.02)
#    print rec.p
#    print rec.q
#    print "Final model:"
#    print np.dot(rec.p,rec.q)
#    print "Final model:"
#    for i in range(data.shape[0]):
#        t = []
#        for j in range(data.shape[1]):
#            t.append( rec.predict(i+1,j+1))
#        print t

    #for factors in [200]:#[10,20,50,100,200]:
    #    cross_validate_movielens_test100k_sivm(1,factors)


    #np.set_printoptions(3, suppress = True)
    #data = generate()
    #rec = SiVMNMFRecommender(data, 1, 3, True)
    #print rec.w.shape
    #print rec.w

    #print rec.h.shape
    #print rec.h

    #print "Factorization"
    #print np.dot(rec.w,rec.h)

    # some performance calculations
    #start_time = time.time()
    #data = load_movielens_ratings100k()
    #rec = SVDSGDRecommender(data, 10, 200, 0.001, 0.0, True, 0.0002, 0.02, False)
    #data = generate()
    #print "Data building took " + str(time.time() - start_time), "seconds"
    #start_time = time.time()
    #data, iterations=5000, factors=2, lr=0.001, reg= 0.02, with_preference=False
    # for time testing :     rec = SVDSGDRecommender(data, 10, 200, 0.001, 0.02, False, 0.001, 0.02, False)

    #print rec.data
    #print "Factorization took " + str(time.time() - start_time), "seconds"
    #print np.dot(rec.p,rec.q)
    #start_time = time.time()
    #eval_movielens_test100k(rec)
    #print "Evaluation took " + str(time.time() - start_time), "seconds"