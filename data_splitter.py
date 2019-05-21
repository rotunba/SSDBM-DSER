"""

"""

import scipy.sparse as sp
import numpy as np
import sys
import random

class DataSplitter(object):

    def __init__(self, datamat, testdatamat, attrmat, k):
        assert sp.isspmatrix_csc(datamat)
        self.datamat = datamat
        self.testdatamat = testdatamat
        self.attrmat = attrmat
        self.k = k
        _, self.num_items = datamat.shape
        assert self.k<=self.num_items
        self.index = [i for i in range(self.num_items)]
        #random.shuffle(self.index)

    def split_data(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(min(self.num_items-base, self.num_items/self.k)):
                tmp.append(self.datamat.getcol(self.index[base+j]))
            base += self.num_items/self.k
            result.append(sp.hstack(tmp,"csc"))
        return result
    
    def split_testdata(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(min(self.num_items-base, self.num_items/self.k)):
                tmp.append(self.testdatamat.getcol(self.index[base+j]))
            base += self.num_items/self.k
            result.append(sp.hstack(tmp,"csc"))
        return result

    def split_attr(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(min(self.num_items-base, self.num_items/self.k)):
                tmp.append(self.attrmat[self.index[base+j]])
            base += self.num_items/self.k
            result.append(np.vstack(tmp))
        return result

