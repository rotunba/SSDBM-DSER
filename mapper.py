"""


Attribute to feature mapping functions for items

map-knn(k) : predict matrix factor H with cosine similarity as weight.
map-lin(alpha, lambda) : linear model train with LeastMSE cost function. 
map-bpr(alpha, lambda) : linear model train with BPR-OPT cost function.
cbf-knn() : k=infinity, predict score directly with sum of cosine similarity.
random() : predict a random set of items as baseline

Inside brackets are parameters needed which can be train via cross-validation.

A, W, H, S matrix might be needed.

"""

import bpr
from math import sqrt, exp, log
import numpy as np
import scipy.sparse as sp
import data_splitter as ds
from copy import copy
import sys
import random
from sklearn.neighbors import NearestNeighbors
import gc
from time import time
import deeprecs as dpr
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(1)

class Mapper(object):

    def __init__(self):
        pass

    def init(self, data, attr, bpr_k=None, bpr_args=None, bpr_model=None):
        assert sp.isspmatrix_csc(data)
        self.data = data
        self.num_users, self.num_items = data.shape
        self.attr = attr
        #assert attr.shape[0] >= self.num_items
        #_, self.num_attrs = attr.shape
        if bpr_model==None:
            self.bpr_k = [self.num_users/5,bpr_k][bpr_k!=None]
            if bpr_args==None:
                self.bpr_args = bpr.BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
            else:
                self.bpr_args = bpr_args
            self.bpr_model = bpr.BPR(self.bpr_k, self.bpr_args)
        else:
            self.bpr_model = bpr_model
            self.bpr_k = bpr_model.D
            self.bpr_args = bpr.BPRArgs(bpr_model.learning_rate, \
                bpr_model.bias_regularization, \
                bpr_model.user_regularization, \
                bpr_model.positive_item_regularization, \
                bpr_model.negative_item_regularization, \
                bpr_model.update_negative_item_factors)
        self.sampler = bpr.UniformUserUniformItem()
    
    def train_init(self):
        tmp = self.data.tocsr()
        self.dataidx = []
        for u in range(self.num_users):
            self.dataidx.append(tmp[u].indices)

    def test_init(self, test_data, test_attr):
        assert sp.isspmatrix_csc(test_data)
        self.num_test_items, _ = test_attr.shape
        tmp = test_data.tocsr()
        self.test_attr = test_attr
        self.test_dataidx = []
        for u in range(self.num_users):
            self.test_dataidx.append(tmp[u].indices)

 
class BPR(Mapper):

    def __init__(self,train_items, train_data, data, attr, bpr_k=None, bpr_args=None, learning_rate=None, penalty_factor=None, path=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor
        self.path = path
        self.train_init()
        #self.mapper_factors = np.random.random_sample((self.bpr_k, self.num_attrs))
        self.bpr_model.init(self.dataidx, self.num_items, None,self.attr)
        self.mapper_bias = np.zeros((self.bpr_k, 1))
        self.train_data  = train_data
        self.train_items = train_items
        
    def get_train_instances(self,train, num_negatives, train_items):
        result = []
        
        for (u,i) in train.keys():
            
            j_array=[]
            for t in xrange(num_negatives):
                j = random.choice(train_items)#np.random.randint(num_items)
                while train.has_key((u, j)) or j in j_array:
                    j = random.choice(train_items)#np.random.randint(num_items)
                j_array.append(j)
                
                result.append([u,i,j])
                
        return result
    
    def train(self,epoch, num_iters):
        train_instances =  self.get_train_instances(self.train_data, 1, self.train_items )
        if (epoch < num_iters):
            self.bpr_model.train(train_instances, self.dataidx, self.num_items, self.sampler, epoch, 0, self.penalty_factor)
        
        return self.bpr_model.user_factors, self.bpr_model.item_factors, self.bpr_model.loss_value

        
    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.bpr_model.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            #it should be ln(1.0/(1.0+exp(-x)) according to thesis)
            #ranking_loss += 1.0/(1.0+exp(x))
            
            try:
                ranking_loss += log(1.0/(1.0+exp(-x)))
            except OverflowError as error:
                # Output expected OverflowErrors.
                #print exp(-x)/(1.0+exp(-x))
                ranking_loss += log(exp(x)/(1.0+exp(x)))
            
#         complexity = 0;
#         for u,i,j in self.loss_samples:
#             complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
#             complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
#             complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
#             complexity += self.bias_regularization * self.item_bias[i]**2
#             complexity += self.bias_regularization * self.item_bias[j]**2

        #XXX: where does 0.5 come from? returns negative BPR-OPT so that it looks we are minimizing it
        #return ranking_loss + 0.5*complexity
        return -ranking_loss
        
    def getGradient(self,x):
        "Numerically-stable sigmoid function."
        try:
            return 1.0/(1.0+exp(x))
        except OverflowError as error:
            # Output expected OverflowErrors.
            #print x
            return exp(-x)/(1.0+exp(-x))
    
    def predict(self, u, i):
        return self.bpr_model.predict(u, i)
        #return self.bpr_model.predict_map(u, i)
        #return np.dot( self.bpr_model.user_factors[u] \
         #  , np.dot(self.mapper_factors, self.attr[i]))  #\
            #+ self.mapper_bias))

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr) 
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i):
        result = []
        i_factors = np.dot(self.mapper_factors, self.test_attr[i])
            #\+self.mapper_bias
        #no i_bias here because we didn't use actual h_i in trainning
        for u in range(self.num_users):
            result.append(np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

    def set_parameter(self, para_set):
        self.learning_rate = para_set[0]
        self.penalty_factor = para_set[1]
