# encoding: utf-8
# filename: profile.py

import cythonFactorize
import numpy as np
from recsys.data.base import load_movielens_ratings100k, generate

import pstats, cProfile

import pyximport
pyximport.install()

data = load_movielens_ratings100k(1, False)
no_users = data.shape[0]
no_items = data.shape[1]
no_ratings = data.size
rowcol = np.array(data.nonzero(),dtype=long)
values = np.array(data.data,dtype=np.float64)
item_indices = np.array(data.indices, dtype = long)     
user_pointers = np.array(data.indptr, dtype = long)
cProfile.runctx("cythonFactorize.svd_plus_plus(\
	no_users, no_items, no_ratings,\
    rowcol, values, item_indices, user_pointers,\
	200, 1, 0.001, 0.0011, 0.001, 0.0011)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()