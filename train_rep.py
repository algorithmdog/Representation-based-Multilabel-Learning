#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]

from latent_factor import *
from arffio        import *
from common        import *
import copy
import logging, Logger
import pickle
import numpy as np
import scipy.sparse as sp
import sampler
import random
import time
from common       import *
from train_common import *

np.random.seed(0)
random.seed(0)

def printUsages():
    print "Usage: train_rep.py [options] train_file model_file"
    print "options"
    print "  -h  hidden_space_dimension: set the hidden space dimension (default 100)"
    print "  -ha hidden_activation: set the hidden activation(default 0)"
    print "       0 -- tanh"
    print "       1 -- linear"
    print "       2 -- relu"
    print "  -oa output_activation: set the output activation(default 0)"
    print "       0 -- sgmoid"
    print "       1 -- linear"
    print "  -l  loss_function: set the loss function(default 0)"
    print "       0 -- negative_log_likelihood"
    print "       1 -- least_sqaure" 
    print "  -l2 l2_regularization: set the l2 regularization(default 0.001)"
    print "  -b  batch_size: set the batch size (default 100)"
    print "  -i  number_of_iter: set the number of iteration(default 10)"
    print "  -st using_sampling: set whether using the sampling scheme(default 1)"
    print "       0 -- not using sampling scheme"
    print "       1 -- using sampling scheme"
    print "  -sr sampling_ratio: set the sampling ratio(default 5)"   
    print "  -sp sparse_threhold: set the threhold (default 0.01)"
    print "  -m  using_external_memory: set using the external memory (default 0). Now you can't use external memory and we will implement the function as soon as possible"
    print "       0 -- not using external memory"
    print "       1 -- using external memory"
    print "  -r  learning_rate: set the learning rate (default 0.001)"


def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file model_file
        printUsages()
        exit(1)

    parameters = copy.deepcopy(rep_default_params)
    parameters["train_file"]   = argv[len(argv) - 2]
    parameters["model_file"]   = argv[len(argv) - 1]

    i = 1
    while i + 1 < len(argv) - 2:
        if  "-h" == argv[i]:
            parameters["h"]     = int(argv[i+1])
        elif "-ha" == argv[i]:
            parameters["ha"]    = ha_map[int(argv[i+1])]
        elif "-oa" == argv[i]:
            parameters["oa"]    = oa_map[int(argv[i+1])]
        elif "-l" == argv[i]:
            parameters["l"]     = lo_map[int(argv[i+1])]
        elif  "-l2" == argv[i]:
            parameters["l2"]    = float(argv[i+1])
        elif "-b" == argv[i]:
            parameters["b"]     = int(argv[i+1]) 
        elif "-i" == argv[i]:
            parameters["i"]     = int(argv[i+1])   
        elif "-st" == argv[i]:
            parameters["st"]    = st_map[int(argv[i+1])];
        elif "-sr" == argv[i]:
            parameters["sr"]    = float(argv[i+1])
        elif "-sp" == argv[i]:
            parameters["sp"]    = float(argv[i+1])
        elif "-m"  == argv[i]:
            parameters["m"]     = m_map[int(argv[i+1])]
        elif "-r"  == argv[i]:
            parameters["r"]     = float(argv[i+1])
        else:
            print argv[i]
            printUsages()
            exit(1)
        i += 2 

    if False == checkParamValid(parameters):
        printUsages()
        exit(1)    

    return parameters



def main(argv):
    parameters = parseParameter(argv)

    train_file  = parameters["train_file"]
    model_file  = parameters["model_file"]

    # read a instance to know the number of features and labels
    train_reader       = SvmReader(train_file, 1)
    x, y, has_next     = train_reader.read()
    parameters["nx"]   = x.shape[1]
    parameters["ny"]   = y.shape[1]
    train_reader.close()

    model        = Model(parameters)
    rater        = AdaGrad(model)
    model.rater  = rater
    thrsel       = ThresholdSel()
    model.thrsel = thrsel

    if m.internal_memory == parameters["m"]:
        model = train_internal(model, train_file, parameters) 
    elif m.external_memory == parameters["m"]:
        model = train_external(model, train_file, parameters)    
    else:
        logger = logging.getLogger(Logger.project_name)
        logger.error("Invalid m param")
        raise Exception("Invalid m param")
    
    #write the model
    #model.clear_for_save()
    model.save(model_file)
    #s = pickle.dumps(model)
    #f = open(model_file, "w")
    #f.write(s)
    #f.close()

if __name__ == "__main__":
    main(sys.argv)
