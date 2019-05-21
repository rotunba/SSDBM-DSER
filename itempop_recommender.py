'''
Random Model.

@author: 
'''

import numpy as np
from evaluate import evaluate_model
from dataset_loader import dataset_loader
from time import time
import argparse
import os

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Random.")
    parser.add_argument('--path', nargs='?', default='data-coldstartitems-itemranking/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--metric', nargs='?', default='mean_squared_error',#binary_crossentropy
                        help='Input data path.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--topk', type=int, default=10,
                        help='TopN')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def run(metric,method,epochs, evaluation_threads, dataset, data_path, start_fold, num_folds,verbose):
    
    all_best_mrr_ten, all_best_ndcg_ten, all_best_mrr,all_best_ndcg,all_best_loss = [], [], [],[], []

    for fold in range(start_fold,num_folds):
        # Loading data
        print("Fold " + str(fold))
        t1 = time()
        path = data_path + "fold" + str(fold) + "/"
        dataset_name = dataset_loader(path, dataset,method)
        training_data_matrix, test_ratings,   test_positives, train_items  = \
        dataset_name.train_matrix, dataset_name.test_ratings, dataset_name.test_positives, dataset_name.train_items
        num_users, num_items = training_data_matrix.shape
        print("Data load done in [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
              %(time()-t1, num_users, num_items, training_data_matrix.nnz, len(test_ratings)))
        
        itempop = training_data_matrix.sum(axis=0)
        
        all_mrr_ten, all_ndcg_ten,all_mrr,all_ndcg = [], [], [],[]
        best_mrr_ten, best_ndcg_ten, best_mrr,best_ndcg, best_loss =  0.0, 0.0, 0.0, 0.0, 123456789
    
        for epoch in xrange(epochs):
    
            # Evaluation
            if epoch %verbose == 0:
                mrr_tens, ndcg_tens,mrrs,ndcgs, losses = evaluate_model(train_items, method,metric, dataset, itempop, test_ratings, test_positives, None, 10, evaluation_threads, None)
                mrr_ten, ndcg_ten,mrr,ndcg,loss = np.array(mrr_tens).mean(), np.array(ndcg_tens).mean(), \
                np.array(mrrs).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                
                all_mrr_ten.append(mrr_ten)
                all_ndcg_ten.append(ndcg_ten)
                
                all_mrr.append(mrr)
                all_ndcg.append(ndcg)
                
                print('Iteration %d: MRR@10 = %.3f, NDCG@10 = %.3f,MRR = %.3f, NDCG = %.3f, LOSS = %.3f' 
                          % (epoch, mrr_ten, ndcg_ten,mrr,ndcg,loss))
                print('AvgMRR@10 = %.3f, AvgNDCG@10 = %.3f,AvgMRR = %.3f, AvgNDCG = %.3f' 
                    %(np.array(all_mrr_ten).mean(), np.array(all_ndcg_ten).mean() , \
                         np.array(all_mrr).mean(), np.array(all_ndcg).mean()))
                if ndcg_ten > best_ndcg_ten:
                    best_itr, best_mrr_ten, best_ndcg_ten, best_mrr,best_ndcg,best_loss = epoch,mrr_ten, ndcg_ten,mrr,ndcg,loss
                    #if args.out > 0:
                        #model.save_weights(model_out_file, overwrite=True)
                
        print("End. Best Iteration %d:  MRR@10 = %.3f, NDCG@10 = %.3f, MRR = %.3f, NDCG = %.3f, LOSS = %.3f. " %(best_itr,best_mrr_ten, best_ndcg_ten,best_mrr,best_ndcg,best_loss))
        all_best_mrr_ten.append(best_mrr_ten)
        all_best_ndcg_ten.append(best_ndcg_ten)
        all_best_mrr.append(best_mrr)
        all_best_ndcg.append(best_ndcg)
        all_best_loss.append(best_loss)
        
    print("End. Mean Scores : MRR@10 = %.3f, NDCG@10 = %.3f, MRR = %.3f, NDCG = %.3f,  LOSS = %.3f. " %(np.array(all_best_mrr_ten).mean(),np.array(all_best_ndcg_ten).mean(),np.array(all_best_mrr).mean(),np.array(all_best_ndcg).mean() ,np.array(all_best_loss).mean()))
    

    