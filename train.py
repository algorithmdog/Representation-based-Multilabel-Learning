#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/utils/Python_Utils")
sys.path.append(path + "/../utils/Python_Utils")

from latent_factor import *
from arffio        import *
import logging, Logger
import pickle
import numpy as np
import scipy.sparse as sp
import sampler
import random
import time

np.random.seed(0)
random.seed(0)

def printUsages():
    print "Usage: train.py [options] train_file model_file"
    print "options"
    print "   -lambda: the regularization coefficient (default 0.001)"
    print "   -struct: the architecture of instance represnation learner: [num_node_layer1,num_node_layer2,...] (default [])"
    print "   -batch: batch, the number of instances in a batch (default 100)"
    print "   -niter: num of iter, the number of iterations (default 20)"
    print "   -num_factor: the number of inner factors"
    print "   -sample_ratio: the ratio of sampling"

def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file model_file
        printUsages()
        exit(1)

    parameters = dict()
    parameters["train_file"]   = argv[len(argv) - 2]
    parameters["model_file"]   = argv[len(argv) - 1]
    parameters["lambda"]       = 0.001
    parameters["struct"]       = [] 
    parameters["batch"]        = 10
    parameters["niter"]        = 20
    parameters["num_factor"]   = 50 
    parameters["sample_ratio"] = 5 

    ##not open option yet
    parameters["mem"]          = 1
    parameters["sample_type"]  = "instance_sample"
    parameters["sparse_thr"]   = 0.01
 
    i = 1
    while i + 1 < len(argv) - 2:
        if  "-lambda" == argv[i]:
            parameters["lambda"]   = float(argv[i+1])
            i += 2
        elif "-struct" == argv[i]:
            line  = argv[i+1]
            line  = line.strip()
            line  = line[1:len(line)-1]
            line  = line.strip()
            if "" != line:
                eles  = line.split(",")
                sizes = map(int, eles)
                parameters["struct"] =  sizes
            i += 2
        elif "-batch" == argv[i]:
            parameters["batch"] = int(argv[i+1]) 
            i += 2
        elif "-niter" == argv[i]:
            parameters["niter"] = int(argv[i+1])   
            i += 2
        elif "-num_factor" == argv[i]:
            parameters["num_factor"] = int(argv[i+1])
            i += 2
        elif "-sample_ratio" == argv[i]:
            parameters["sample_ratio"] = float(argv[i+1])
            i += 2
        else:
            print argv[i]
            printUsages()
            exit(1)
        

    return parameters

def train_mem(train_file, parameters):
    model  = Model(parameters)
    batch  = parameters["batch"]
    niter  = parameters["niter"]
    sample = sampler.get_sample(parameters)   
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
    batch  = parameters["batch"]
    niter  = parameters["niter"]
    sample = sampler.get_sample(parameters)
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
    train_reader = SvmReader(train_file, 1)
    x, y, has_next = train_reader.read()
    parameters["num_feature"] = x.shape[1]
    parameters["num_label"]   = y.shape[1]
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
