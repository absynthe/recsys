"""
Base I/O code for all datasets
"""

# Authors: Anamaria Todor <anamaria@gameanalytics.com>

from os.path import dirname
from os.path import join
import numpy as np
import scipy.sparse as sparse
import time
import math
from recsys.recommenders.SVDSGDRecommender import SVDSGDRecommender

def generate():
    print "Building user ratings matrix"
    R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

    R = sparse.lil_matrix(R,dtype=np.float64)
    return R.tocsr()

def load_movielens_ratings100k(load_timestamp=False):
    """ Load and return a sparse matrix of the 100k ratings MovieLens dataset
        (only the user ids, item ids and ratings).
        Optional:  timestamps

    @type load_timestamp - bool, optional (default = False)
    @param load_timestamp - whether it loads the timestamp

    @rtype: sparse.csr_matrix
    @return: the user ratings matrix model

    """
    base_dir = join(dirname(__file__), 'raw/ml-100k/')
    #Read data
    data_info = np.loadtxt(base_dir + 'u.info', delimiter=' ', dtype={'names': ('amount', 'what'),
                      'formats': ('i4', 'S1')})
    no_users = data_info[0][0]
    no_items = data_info[1][0]
    data_ratings = np.loadtxt(base_dir + 'u1.base', delimiter='\t', usecols=(0, 1, 2), dtype=int)
    if load_timestamp:
        #TODO think if you want to do anything with timestamps
        return None
    else:
        ratings_matrix = sparse.lil_matrix((no_users,no_items), dtype=np.float64)
        for user_id, item_id, rating in data_ratings:
            ratings_matrix[user_id-1, item_id-1] = np.float64(rating)
        return ratings_matrix.tocsr()

def eval_movielens_test100k(rec, load_timestamp=False):
    base_dir = join(dirname(__file__), 'raw/ml-100k/')
    test_ratings = np.loadtxt(base_dir + 'u1.test', delimiter='\t', usecols=(0, 1, 2), dtype=int)
    total = 0
    n = 0
    for user_id, item_id, rating in test_ratings:
        estimate = rec.predict(user_id,item_id)
        total += (rating-estimate) **2
        n+=1
        print rating, estimate
    print "Rmse is: " + str(math.sqrt(total/n))



def load_movielens_titles100k():
    """ Load and return a dictionary of the movie titles in the
        100k ratings MovieLens dataset

        @rtype: dict
        @return: a dictionary which has movies ids as keys and movie titles as values
    """
    base_dir = join(dirname(__file__), 'raw/ml-100k/')
    data_titles = np.loadtxt(base_dir + 'u.item',
             delimiter='|', usecols=(0, 1), dtype=str)
    data_t = []
    for item_id, label in data_titles:
        data_t.append((int(item_id), label))
    movie_titles = dict(data_t)
    return movie_titles

def load_netflix_r(load_timestamp=False):
    """ Load and return a sparse matrix of the full Netflix dataset
        (only the user ids, item ids and ratings).
        Optional:  timestamps

        @type load_timestamp - bool, optional (default = False)
        @param load_timestamp - whether it loads the timestamp

        @rtype: sparse.csr_matrix , dict
        @return: the user ratings matrix model, together with the user_id mappings to matrix ids
    """

    base_dir = join(dirname(__file__), 'raw/netflix/training_set/mv_')

    no_items = 17770
    no_users = 2649429#480189
    id_count = 0

    ratings_matrix = sparse.lil_matrix((no_users,no_items), dtype=np.float64)
    user_id_to_id = {}
    for item_id in range(1,no_items+1):
        f = open(base_dir+str(item_id).zfill(7)+".txt", 'r')
        content = f.readlines()

        for line in content[1:]:        #ignore first line
            user_id, rating, date=line.split(',')
            ratings_matrix[int(user_id)-1,item_id-1] = rating
            # look for user_id in dictionary
            stored_user_id = user_id_to_id.get(user_id)
            if stored_user_id is not None:
                #update ratings matrix directly if it's there
                ratings_matrix[stored_user_id,item_id-1] = rating
            else:
                #otherwise first save it in dict
                user_id_to_id[user_id]=id_count
                ratings_matrix[id_count,item_id-1] = rating
                id_count+=1
    # convert matrix to CSR
    return ratings_matrix.tocsr()#, user_id_to_id

def load_netflix_t():
    """ Load and return a dictionary of the movie titles in the
        whole Netflix dataset

        @rtype: dict
        @return: a dictionary which has movies ids as keys and movie titles as values
    """
    base_dir = join(dirname(__file__), 'raw/netflix')
    data_titles = np.loadtxt(base_dir + 'movie_titles.txt',
             delimiter=',', usecols=(0, 2), dtype=str)
    data_t = []
    for item_id, label in data_titles:
        data_t.append((int(item_id), label))
    movie_titles = dict(data_t)
    return movie_titles

if __name__ == "__main__":
    start_time = time.time()
    model = load_netflix_r()
    print (time.time() - start_time)/60.00, "minutes"