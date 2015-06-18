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

def printUsages():
    print "Usage: train.py [options] train_file model_file"
    print "options"
    print "   -m: training with all data in mem (default 0)"
    print "         0, a part of training data in mem"
    print "         1, all data in mem, 0 denote sgd"
    print "   -i: ins lambda, the instance regularization coefficient (default 0.001)"
    print "   -l: label lambda, the label regularization coefficient (default 0.001)" 
    print "   -s: sizes, the architecture: [num_node_layer1,num_node_layer2,...] (default [])"
    print "   -b: batch, the number of instances in a batch (default 100)"
    print "   -n: num of iter, the number of iterations (default 20)"
    print "   -t: sample_type, the sample_type"
    print "         instance_sample, instance orient sampling scheme"
    print "         label_sample, label orient sampling scheme"
    print "         correlation_sample, sampling scheme exploits label correlations"
    print "   -num_factor: the number of inner factors"

def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file model_file
        printUsages()
        exit(1)

    parameters = dict()
    parameters["train_file"]   = argv[len(argv) - 2]
    parameters["model_file"]   = argv[len(argv) - 1]
    parameters["ins_lambda"]   = 0.001
    parameters["label_lambda"] = 0.001
    parameters["sizes"]        = [] 
    parameters["batch"]        = 10
    parameters["niter"]        = 20
    parameters["sample_type"]  = "full"
    parameters["mem"]          = 1
   
    i = 1
    while i + 1 < len(argv) - 2:
        if  "-i" == argv[i]:
            parameters["ins_lambda"]   = float(argv[i+1])
        elif "-l" == argv[i]:
            parameters["label_lambda"] = float(argv[i+1])
        elif "-s" == argv[i]:
            line  = argv[i+1]
            line  = line.strip()
            line  = line[1:len(line)-1]
            line  = line.strip()
            if "" != line:
                eles  = line.split(",")
                sizes = map(int, eles)
                parameters["sizes"] =  sizes
        elif "-b" == argv[i]:
            parameters["batch"] = int(argv[i+1]) 
        elif "-n" == argv[i]:
            parameters["niter"] = int(argv[i+1])   
        elif "-t" == argv[i]:
            parameters["sample_type"] = argv[i+1]
        elif "-m" == argv[i]:
            parameters["mem"] = int(argv[i+1])
        elif "-sample_ratio" == argv[i]:
            parameters["sample_ratio"] = int(argv[i+1])
        elif "-num_factor" == argv[i]:
            parameters["num_factor"] = int(argv[i+1])
        else:
            printUsages()
            exit(1)
        i += 2

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

    for iter1 in xrange(niter):
        start = 0
        end = batch
        while start < num:
            logger.info("start = %d, end = %d\n"%(start, end))
            if end > num:   end = num

            batch_x = x[start:end, :]
            batch_y = y[start:end, :] 
            batch_i = sample.sample(batch_y)
            model.update(batch_x, batch_y, batch_i)      

            start += batch;
            end += batch;

        logger.info("The %d-th iteration completes"%(iter1+1)); 
    
    #####tuning the threshold
    start = 0
    end = batch
    while start < num:
        if end > num: end = num
        batch_x = x[start:end,:]
        batch_y = y[start:end,:]
        batch_p = model.ff(batch_x)
        model.thrsel.update(batch_p, batch_y)
        start += batch
        end   += batch
    logger.info("The threshold tuning completes") 

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
    sample_type = parameters["sample_type"];
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
    s = pickle.dumps(model)
    f = open(model_file, "w")
    f.write(s)
    f.close()

if __name__ == "__main__":
    main(sys.argv)
