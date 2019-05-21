"""


Main program to perform training, with a splitter to do k-fold cross-validation

"""
import mapper
import data_splitter as ds
from bpr import BPRArgs
import numpy as np
import scipy.sparse as sp

from evaluate import evaluate_model
from dataset_loader import dataset_loader
from time import time
import multiprocessing as mp

#do this for consistent results
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

def run(metric,learning_rate,bpr_k,epochs, evaluation_threads, dataset, data_path, start_fold, num_folds,verbose):
    #all parameters needed setting are here
    penalty_factor = 0.0001
    bpr_args = BPRArgs(learning_rate, 1, 0.01, 0.01, 0.01)
    
    all_best_mrr_ten, all_best_ndcg_ten, all_best_mrr,all_best_ndcg,all_best_loss = [], [], [],[], []

    for fold in range(start_fold,num_folds):
        # Loading data
        print("Fold " + str(fold))
        t1 = time()
        path = data_path + "fold" + str(fold) + "/"
        dataset_name = dataset_loader(path, dataset,"map-bpr") 
                
        training_data_matrix, test_ratings, test_data_matrix,  test_positives,  train_items  = \
        dataset_name.train_matrix, dataset_name.test_ratings,dataset_name.test_matrix, dataset_name.test_positives, dataset_name.train_items
        
        training_data = sp.csc_matrix(training_data_matrix)  
        test_data = sp.csc_matrix(test_data_matrix)
        
        attr = []
       
        splitter = ds.DataSplitter(training_data,test_data, attr, 1)
        data_matrix = splitter.split_data()
        attr_matix = None#splitter.split_attr()
    
        
        num_users, num_items = training_data.shape
        print("Data load done in [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
              %(time()-t1, num_users, num_items, training_data.nnz, len(test_ratings)))
             
        model = mapper.BPR(train_items,training_data_matrix,sp.hstack(data_matrix,"csc"), None, bpr_k, bpr_args, learning_rate, penalty_factor, path)   
                    
   
        all_mrr_ten, all_ndcg_ten,all_mrr,all_ndcg = [], [], [],[]
        best_mrr_ten, best_ndcg_ten, best_mrr,best_ndcg, best_loss =  0.0, 0.0, 0.0, 0.0, 123456789
    
        for it in range(epochs): 
            bpr_users, bpr_items, loss = model.train(it,epochs) 
            # Evaluation
            if it %verbose == 0:
                mrr_tens, ndcg_tens,mrrs,ndcgs, losses = evaluate_model(train_items,"bpr",metric, dataset, model, test_ratings, test_positives, None, 10, evaluation_threads, None)
                mrr_ten, ndcg_ten,mrr,ndcg,loss = np.array(mrr_tens).mean(), np.array(ndcg_tens).mean(), \
                np.array(mrrs).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                
                all_mrr_ten.append(mrr_ten)
                all_ndcg_ten.append(ndcg_ten)
                
                all_mrr.append(mrr)
                all_ndcg.append(ndcg)
                
                print('Iteration %d: MRR@10 = %.3f, NDCG@10 = %.3f,MRR = %.3f, NDCG = %.3f, LOSS = %.3f' 
                          % (it, mrr_ten, ndcg_ten,mrr,ndcg,loss))
                print('AvgMRR@10 = %.3f, AvgNDCG@10 = %.3f,AvgMRR = %.3f, AvgNDCG = %.3f' 
                    %(np.array(all_mrr_ten).mean(), np.array(all_ndcg_ten).mean() , \
                         np.array(all_mrr).mean(), np.array(all_ndcg).mean()))
                if ndcg_ten > best_ndcg_ten:
                    best_itr, best_mrr_ten, best_ndcg_ten, best_mrr,best_ndcg,best_loss = it,mrr_ten, ndcg_ten,mrr,ndcg,loss
                    #if args.out > 0:
                        #model.save_weights(model_out_file, overwrite=True)
                
        print("End. Best Iteration %d:  MRR@10 = %.3f, NDCG@10 = %.3f, MRR = %.3f, NDCG = %.3f, LOSS = %.3f. " %(best_itr,best_mrr_ten, best_ndcg_ten,best_mrr,best_ndcg,best_loss))
        all_best_mrr_ten.append(best_mrr_ten)
        all_best_ndcg_ten.append(best_ndcg_ten)
        all_best_mrr.append(best_mrr)
        all_best_ndcg.append(best_ndcg)
        all_best_loss.append(best_loss)
        
    print("End. Mean Scores : MRR@10 = %.3f, NDCG@10 = %.3f, MRR = %.3f, NDCG = %.3f,  LOSS = %.3f. " %(np.array(all_best_mrr_ten).mean(),np.array(all_best_ndcg_ten).mean(),np.array(all_best_mrr).mean(),np.array(all_best_ndcg).mean() ,np.array(all_best_loss).mean()))
