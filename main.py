'''
DSER and other recommendation models.

@author: 
'''
import logging
from keras import backend as K
import argparse
import multiprocessing as mp
import os
import bpr_recommender
import itempop_recommender
import deeprecs
#do this for consistent results
from numpy.random import seed
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
    parser = argparse.ArgumentParser(description="Run recommender systems.")
    parser.add_argument('--method', nargs='?', default='dser',
                        help='Recommender system to use.')
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--metric', nargs='?', default='mean_squared_error',#binary_crossentropy
                        help='Metric.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size.')
    parser.add_argument('--start_fold', type=int, default=0,
                        help='Start Fold.')
    parser.add_argument('--layers', nargs='?', default='[64,64]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--topk', type=int, default=10,
                        help='TopN')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--num_eval_threads', type=int, default=1,
                        help='Number of evaluation threads')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--decay', type=float, default=0.00000,
                        help='learning rate decay')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd, categorical_adam')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    return parser.parse_args()


if __name__ == '__main__':
    set_keras_backend("theano")
    logging.getLogger("theano.gof.compilelock").setLevel(logging.CRITICAL)

    args = parse_args()

    path = args.path
    dataset = os.getenv("dataset", args.dataset)
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    learner = args.learner
    learning_rate = float(os.getenv("lr", args.lr))
    batch_size = int(os.getenv("batch_size", args.batch_size))
    epochs = int(os.getenv("epochs", args.epochs))
    verbose = args.verbose
    decay = args.decay
    topK = args.topk
    evaluation_threads = int(os.getenv("num_eval_threads", mp.cpu_count()))
    metric = os.getenv("metric", args.metric)
    start_fold = int(os.getenv("start_fold", args.start_fold))
    method = os.getenv("method", args.method)
    num_negatives = int(os.getenv("num_neg", args.num_neg))
    dropout_rate = float(os.getenv("dropout_rate", args.dropout_rate))
    
    print("Metric: %s " %(metric))
    print("Batch size: %s " %(batch_size))
    print("Learning Rate: %s " %(learning_rate))
    print("Dataset: %s " %(dataset))
    print("Evaluation Threads: %s " %(evaluation_threads))
    print("#Epochs: %s " %(epochs))
    num_folds = 3
    
    if (method == 'bpr'):
        print 'USING BPR' 
        bpr_recommender.run(metric,learning_rate, layers[0], epochs, evaluation_threads, dataset, path, start_fold, num_folds,verbose)
    elif (method == 'itempop'):
        print 'USING ItemPop' #use 1 epoch since itempop does not need training     
        itempop_recommender.run(metric, method, 1, evaluation_threads, dataset, path, start_fold, num_folds,verbose)
    elif (method == 'neumf'):
        print 'USING NeuMF'     
        deeprecs.run(num_negatives, metric,method, layers,reg_layers, decay, batch_size, learning_rate, epochs, evaluation_threads, dataset, path, start_fold, num_folds,verbose)
    elif (method == 'dser'):
        print 'USING DSER'     
        deeprecs.run(dropout_rate, num_negatives,metric,method, layers,reg_layers, decay, batch_size, learning_rate, epochs, evaluation_threads, dataset, path, start_fold, num_folds,verbose)