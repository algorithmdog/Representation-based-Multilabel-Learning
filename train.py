#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]

from latent_factor import *
from arffio        import *
import logging, Logger
import pickle
import numpy as np
import scipy.sparse as sp
import sampler
import random
import time
from common import *

np.random.seed(0)
random.seed(0)

def printUsages():
    print "Usage: train.py [options] train_file model_file"
    print "options"
    print "  -h  hidden_space_dimension: set the hidden space dimension (default 100)"
    print "  -ha hidden_activation: set the hidden activation(default 0)"
    print "       0 -- tanh"
    print "       1 -- linear"
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

def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file model_file
        printUsages()
        exit(1)

    parameters = dict()
    parameters["train_file"]   = argv[len(argv) - 2]
    parameters["model_file"]   = argv[len(argv) - 1]
    #structure parameter
    parameters["h"]            = 100
    parameters["ha"]           = act.tanh
    parameters["oa"]           = act.sgmoid
   
    #training parameter
    parameters["l"]            = lo.negative_log_likelihood
    parameters["l2"]           = 0.001
    parameters["b"]            = 10
    parameters["i"]            = 20
    parameters["st"]           = st.instance_sampler
    parameters["sr"]           = 5 
    parameters["sp"]           = 0.01
 
    parameters["sizes"]        = []
    parameters["mem"]          = 1


    i = 1
    while i + 1 < len(argv) - 2:
        if  "-h" == argv[i]:
            parameters["h"]     = int(argv[i+1])
        elif "-ha" == argv[i]:
            parameters["ha"]    = ha_map[int(argv[i+1])]
        elif "-oa" == argv[i]:
            parameters["oa"]    = oa_map[int(argv[i+1])]
            print parameters["oa"]
        elif "-l" == argv[i]:
            parameters["l"]     = lo_map[int(argv[i+1])]
        elif  "-l2" == argv[i]:
            parameters["l2"]    = float(argv[i+1])
        elif "-b" == argv[i]:
            parameters["b"]     = int(argv[i+1]) 
        elif "-i" == argv[i]:
            parameters["i"]     = int(argv[i+1])   
        elif "-st" == argv[i]:
            parameters["st"]    = int(argv[i+1]);
        elif "-sr" == argv[i]:
            parameters["sr"]    = float(argv[i+1])
        elif "-sp" == argv[i]:
            parameters["sp"]    = float(argv[i+1])
        else:
            print argv[i]
            printUsages()
            exit(1)
        i += 2 

    return parameters

def train_mem(train_file, parameters):
    model  = Model(parameters)
    batch  = parameters["b"]
    niter  = parameters["i"]
    sample = sampler.get_sampler(parameters)   
    logger = logging.getLogger(Logger.project_name)
    logger.info("Model initialization done")

    train_reader = SvmReader(train_file)
    x, y = train_reader.full_read()
    num, _ = y.shape
    #if None == sample: idx_y = sp.csr_matrix(np.ones(y.shape))
    #else: idx_y = sp.csr_matrix(sample.sample(y))
    logger.info("Training data loading done")

    sample.update(y)
    logger.info("Sampling initialization done")

    start_time = time.time()
    for iter1 in xrange(niter):
        start = 0
        end = batch
        while start < num:
            #logger.info("start = %d, end = %d\n"%(start, end))
            if end > num:   end = num
            
#            import cProfile, pstats, StringIO
#            pr =  cProfile.Profile()
#            pr.enable()

            batch_x = x[start:end, :]
            batch_y = y[start:end, :] 
            batch_i = sample.sample(batch_y)
            model.update(batch_x, batch_y, batch_i)      

            start += batch;
            end += batch;
#            pr.disable()
#            s = StringIO.StringIO()
#            sortby = 'cumulative'
#            ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
#            ps.print_stats()
#            print "update",s.getvalue()
            
        logger.info("The %d-th iteration completes"%(iter1+1)); 
    
    #####tuning the threshold
    total = 0
    start = 0
    end = batch
    while start < num and total < 1000:
        if end > num: end = num
        batch_x = x[start:end,:]
        batch_y = y[start:end,:]
        batch_p = model.ff(batch_x)
        model.thrsel.update(batch_p, batch_y)
        start += batch
        end   += batch
        total += 1

    logger.info("The threshold tuning completes") 
    end_time = time.time()
    logger.info("The training time is %f"%(end_time-start_time))

    return model


def train(train_file, parameters):
    model  = Model(parameters)
    batch  = parameters["b"]
    niter  = parameters["i"]
    sample = sampler.get_sampler(parameters)
    logger = logging.getLogger(Logger.project_name)


    ##initilization the sampler
    train_reader   = SvmReader(train_file, batch)
    has_next       = True
    while has_next:
        x,y,has_next = train_reader.read()
        sample.update(y)
    
    ##weight updates
    for iter1 in xrange(niter): 
        train_reader = SvmReader(train_file, batch)
    
        has_next = True
        while has_next:
            x, y, has_next = train_reader.read()
            idx            = sample.sample(y)
            model.update(x, y, idx)

        logger.info("The %d-th iteration completes"%(iter1+1)); 
        train_reader.close()

    ##tuning threshold
    train_reader = SvmReader(train_file, batch)
    x, y, has_next = train_reader.read()
    while has_next:
        p = model.ff(x)
        model.thrsel.update(p, y)
        x, y, has_next = train_reader.read()
        

    return model


def main(argv):
    parameters = parseParameter(argv)

    train_file  = parameters["train_file"]
    model_file  = parameters["model_file"]
    mem         = parameters["mem"]

    # read a instance to know the number of features and labels
    train_reader       = SvmReader(train_file, 1)
    x, y, has_next     = train_reader.read()
    parameters["nx"]   = x.shape[1]
    parameters["ny"]   = y.shape[1]
    train_reader.close()

    if 1 == mem:
        model = train_mem(train_file, parameters) 
    else:
        model = train(train_file, parameters)    

    
    #write the model
    #model.clear_for_save()
    model.save(model_file)
    #s = pickle.dumps(model)
    #f = open(model_file, "w")
    #f.write(s)
    #f.close()

if __name__ == "__main__":
    main(sys.argv)
