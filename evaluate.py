'''
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and AUC
    
@author: 
'''
import math
import random
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_metric = None
_method = None
_dataset = None
_model = None
_testRatings = None
_testPositives = None
_itemAttributes = None
_K = None
_bpr_users = None
_train_items = None

def evaluate_model(train_items,method,metric, dataset, model, testRatings, testPositives, itemAttributes, K, num_thread, bpr_users):
    """
    Evaluate the performance (Hit_Ratio of top-K recommendation), AUC
    Return: score of each test rating.
    """
    global _train_items
    global _metric
    global _method
    global _dataset
    global _model
    global _cold_start
    global _testRatings
    global _testPositives
    global _itemAttributes
    global _K
    global _bpr_users
    
    _metric = metric
    _method = method
    _dataset = dataset
    _model = model
    _testRatings = testRatings
    _testPositives = testPositives
    _itemAttributes = itemAttributes
    _K = K
    _bpr_users = bpr_users
    _train_items = train_items
    
    auc_tens,mrr_tens, ndcg_tens, recall_tens, aucs,mrrs, ndcgs, recalls,losses = [],[],[],[],[],[],[],[],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        auc_tens = [r[0] for r in res]
        mrr_tens = [r[1] for r in res]
        ndcg_tens = [r[2] for r in res]
        recall_tens = [r[3] for r in res]
        aucs = [r[4] for r in res]
        mrrs = [r[5] for r in res]
        ndcgs = [r[6] for r in res]
        recalls = [r[7] for r in res]
        losses = [r[8] for r in res]
        return (mrr_tens, ndcg_tens,mrrs,ndcgs,losses)
    # Single thread
    for idx in xrange(len(_testRatings)):
        (auc_ten,mrr_ten, ndcg_ten, recall_ten, auc,mrr, ndcg, recall,loss) = eval_one_rating(idx)
        auc_tens.append(auc_ten)
        mrr_tens.append(mrr_ten)
        ndcg_tens.append(ndcg_ten)
        recall_tens.append(recall_ten)
        aucs.append(auc)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        recalls.append(recall)
        losses.append(loss)
    return (mrr_tens, ndcg_tens,mrrs,ndcgs,losses)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items_positives = _testPositives[rating[0]]
    items = []
    bpr_users = []
    bpr_items = []
    features = []
    users = []
    gtItem = rating[1]
    predicted_rating = 0
    #get 99 negatives
    while True:
        t = random.choice(_train_items)
        if t in items_positives or t == gtItem:# or t > 785 or t == 90 -> rate only among unseen items
            continue
        items.append(t)
        if len(items) == 99:
            break
        
    items.append(gtItem)
    
    # Get prediction scores
    map_item_score = {}
    predictions = {}
    
    if (_model is not None):
        u = rating[0]   
        
        #for each genre of each item in items, add the item and the genre
        #for each item in items
        for item in items:
            users.append(u)
        
        predictions = {}
        
        if _method == 'itempop':
            for i in xrange(len(items)):
                item = items[i]
                predictions[i] = _model[0,item]
        elif (_method == 'bpr'): 
            
            for i in xrange(len(items)):
                predictions[i] = _model.predict(u,items[i])
        else:
            predictions = _model.predict([np.array(users),np.array(items)], 
                                      verbose=0, batch_size=len(items))
#             predictions = _model.predict([np.array(users), np.array(items),  np.array(features)], 
#                                       verbose=0)#batch_size=50,
        
        for i in xrange(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        #items.pop() 
    else:
        for i in xrange(len(items)):
            predictions[i] = random.uniform(0, 1)
            map_item_score[items[i]] = predictions[i]
        #items.pop()
    #print  map_item_score
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    ranklist_all = heapq.nlargest(len(items), map_item_score, key=map_item_score.get)
    auc_ten = getAuc(ranklist, gtItem)
    mrr_ten = getMRR(ranklist, gtItem)
    ndcg_ten = getNDCG(ranklist, gtItem)
    recall_ten = getRecall(ranklist, gtItem)
    auc = getAuc(ranklist_all, gtItem)
    mrr = getMRR(ranklist_all, gtItem)
    ndcg = getNDCG(ranklist_all, gtItem)
    recall = getRecall(ranklist_all, gtItem)
    if _metric == "binary_crossentropy":
        loss = getEntropies(ranklist, predictions)
    else:
        loss = getMSE(ranklist, predictions)
    return (auc_ten,mrr_ten, ndcg_ten, recall_ten, auc,mrr, ndcg, recall,loss)

def get_attributes_per_item(item):
    return _itemAttributes[item]

def getMSE(ranklist, predictions):
    mses = []
    for i in xrange(len(ranklist)-1):
        mses.append(math.pow(predictions[i], 2))
    #process final pair
    mses.append(math.pow((1 - predictions[i+1]), 2))
    return np.array(mses).mean()

def getEntropies(ranklist, predictions):
    entropies = []
    for i in xrange(len(ranklist)-1):
        entropies.append(CrossEntropy(0, predictions[i]) )
    #process final pair
    entropies.append(CrossEntropy(1, predictions[i+1]) )
    return np.array(entropies).mean()

def CrossEntropy(y, yHat):
    if y == 1:
      return -math.log(yHat)
    else:
      return -math.log(1 - yHat)
  
def getRecall(ranklist, gtItem):
    
    for i in xrange(len(ranklist)-1):
        item = ranklist[i]
        if item == gtItem:
            return 1
    return 0

def getMRR(ranklist, gtItem):
    
    for i in xrange(len(ranklist)-1):
        item = ranklist[i]
        if item == gtItem:
            return 1/float(i + 1)
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)-1):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / float(math.log(i+2))
    return 0

def getAuc(ranklist, gtItem):
    result = len(ranklist)
    negativeItemsCount = len(ranklist) - 1
    
    for i in xrange(len(ranklist)):
        result = result - 1
        item = ranklist[i]
        if item == gtItem:
            break
            
    return result/float(negativeItemsCount)