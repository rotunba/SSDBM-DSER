'''
DPR Model.

@author: 
'''
import numpy as np
from keras import backend as K
#from keras import initializations
from keras.regularizers import l2
from keras.models import  Model

from keras.layers import Embedding, Input, Dense, Concatenate, Multiply, Reshape, Flatten, Dropout, Dot
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model, get_attributes_per_item
from dataset_loader import dataset_loader
from time import time
import sys
import argparse
import multiprocessing as mp
import random
import os

#do this for consistent results
from numpy.random import seed


#from compiler.ast import Mul
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
        
#4: random initializations
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run neurec_c_no_bpr.")
    parser.add_argument('--method', nargs='?', default='map-bpr',
                        help='Recommender system to use.')
    parser.add_argument('--path', nargs='?', default='data-coldstartitems-itemranking/',
                        help='Input data path.')
    parser.add_argument('--metric', nargs='?', default='mean_squared_error',#binary_crossentropy
                        help='Metric.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size.')
    parser.add_argument('--start_fold', type=int, default=0,
                        help='Start Fold.')
    parser.add_argument('--layers', nargs='?', default='[19,19]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--topk', type=int, default=10,
                        help='TopN')
    parser.add_argument('--decay', type=float, default=0.00000,
                        help='learning rate decay')
    parser.add_argument('--lr', type=float, default=0.008,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd, categorical_adam')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def deeprec_neumf(num_users,num_items, layers = [20,10], reg_layers=[0,0] ):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0], name = 'mf_embedding_user',
                                   embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0], name = 'mf_embedding_item',
                                   embeddings_regularizer = l2(reg_layers[0]), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = "mlp_embedding_user",
                                   embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'mlp_embedding_item',
                                   embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = Concatenate()([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
    return model

def deeprec_dser(dropout_rate,num_users, num_items, layers = [20,10], reg_layers=[0,0] ):
    num_layer = len(layers) #Number of layers in the MLP
    assert num_layer == len(reg_layers)
    
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input') 
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input') 
 
        
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0], name = 'user_embedding',
                                  embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0], name = 'item_embedding',
                                  embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    idx = 0

    for idx in xrange(1, num_layer):
        user_latent = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='tanh', name = 'loweruser%d' %(idx+1))(user_latent)
        item_latent = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='tanh', name = 'loweritem%d' %idx)(item_latent)
        
    dpr_vector = Multiply()([user_latent,item_latent])
    #dpr_vector = Concatenate()([dpr_vector,item_features_input])
    
    dpr_vector = Dropout(dropout_rate)(dpr_vector)
     
    dpr_vector = Dense(64, kernel_regularizer= l2(0), activation='tanh', name = 'deeprec-e%d' %(idx))(dpr_vector)
    
    prediction = Dense(1, activation='tanh',  name = "prediction")(dpr_vector)
    #prediction = Dot(axes=1)([user_latent,item_features_latent])   
    model = Model(inputs=[user_input,item_input ], 
                  outputs=prediction)
    
    return model


def get_train_instances(train, train_items, num_negatives):
    user_input, item_input, labels = [],[],[]
    for (u,i) in train.keys():

        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        # negative instances
        j_array=[]
        for t in xrange(num_negatives):
            j = random.choice(train_items)#np.random.randint(num_items)
            while train.has_key((u, j)) or j in j_array:
                j = random.choice(train_items)#np.random.randint(num_items)
            j_array.append(j)
             
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
        
             
    return user_input, item_input, labels

def run(dropout_rate, num_negatives , metric,method, layers, reg_layers, decay, batch_size,learning_rate,epochs, evaluation_threads, dataset, data_path, start_fold, num_folds,verbose):
    all_best_mrr_ten, all_best_ndcg_ten, all_best_mrr,all_best_ndcg,all_best_loss = [], [], [],[], []
    
    for fold in range(start_fold,num_folds):
        learning_rate = float(os.getenv("lr", learning_rate))
        print("Fold " + str(fold))
        t1 = time()
        path = data_path + "fold" + str(fold) + "/"
        dataset_name = dataset_loader(path, dataset, method)

        training_data_matrix, test_ratings,   test_positives, itemAttributes, train_items  = \
        dataset_name.train_matrix, dataset_name.test_ratings, dataset_name.test_positives, \
        dataset_name.item_features, dataset_name.train_items
        
        
        num_users, num_items = training_data_matrix.shape
        print("Data load done in [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
              %(time()-t1, num_users, num_items, training_data_matrix.nnz, len(test_ratings)))
        
        
        if (method == "neumf"):
            model = deeprec_neumf(num_users, num_items, layers, reg_layers)
        elif (method == "dser"):
            model = deeprec_dser(dropout_rate, num_users, num_items, layers, reg_layers)
        
                    
        model.compile(optimizer=Adam(lr=learning_rate, decay=decay), loss=metric)#mean_squared_error
              
        all_mrr_ten, all_ndcg_ten,all_mrr,all_ndcg = [], [], [],[]
        best_mrr_ten, best_ndcg_ten, best_mrr,best_ndcg, best_loss =  0.0, 0.0, 0.0, 0.0, 123456789
    
        for epoch in xrange(epochs):
            user_input, item_input, labels = \
            get_train_instances(training_data_matrix,  train_items,num_negatives)
            
            #np.array(user_gender_input),np.array(user_age_input),
            model.fit([np.array(user_input),np.array(item_input)], #input
            np.array(labels), # labels
            batch_size=batch_size, validation_split=0.00,epochs=1, verbose=0, shuffle=True)#validation_split=0.30,
            
            # Evaluation
            if epoch %verbose == 0:
                mrr_tens, ndcg_tens,mrrs,ndcgs, losses = evaluate_model(train_items,method,metric,dataset, model, test_ratings, test_positives, itemAttributes, 10, evaluation_threads, None)
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
