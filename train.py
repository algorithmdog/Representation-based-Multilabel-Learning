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
    print "         full, full matrix"
    print "         instance_sample, instance orient sampling scheme"
    print "         label_sample, label orient sampling scheme"
    print "         correlation_sample, sampling scheme exploits label correlations"


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
    parameters["batch"]        = 100
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
            parameters["mem"]  = int(argv[i+1])
        else:
            printUsages()
            exit(1)
        i += 2

    return parameters

def train_mem(train_file, parameters, sample = None):
    model = Model(parameters)
    batch = parameters["batch"]
    niter = parameters["niter"]
    
    logger = logging.getLogger(Logger.project_name)
    logger.info("The latent_factor model starts")

    train_reader = ArffReader(train_file)
    x,y = train_reader.full_read_sparse()
    num, _ = y.shape
    for iter1 in xrange(niter):
        start = 0
        end = batch
        while start < num:
            if end > num:   end = num

            batch_x = x[start:end, :]
            batch_y = y[start:end, :] 
            if None == sample:  idx_y = np.ones(batch_y.shape)
            else:   idx_y = sample.sample(batch_y)
            model.update(batch_x, batch_y, idx_y)      

            start += batch;
            end += batch;

        logger.info("The %d-th iteration completes"%(iter1+1));

    return model


def train(train_file, parameters, sample = None):
    model = Model(parameters)
    batch = parameters["batch"]
    niter = parameters["niter"]

    logger = logging.getLogger(Logger.project_name)
    logger.info("The latent_factor model starts")

    for iter1 in xrange(niter): 
        train_reader = ArffReader(train_file, batch)

        #idx_file specified
        if None != sample:

            x, y, has_next = train_reader.read_sparse()
            idx            = sample.sample(y)
            while has_next:
                model.update(x, y, idx)
                x, y, has_next = train_reader.read_sparse()
                idx            = sample.sample(y)

        #idx_file not specified, full
        else:
            x, y, has_next = train_reader.read_sparse()
            idx            = np.ones(y.shape)

            while has_next:
                model.update(x, y, idx)    
                x, y, has_next = train_reader.read_sparse()
                idx = np.ones(y.shape)   

        logger.info("The %d-th iteration completes"%(iter1+1)); 
        train_reader.close()

    logger.info("The latent_factor model completes")

    return model


def main(argv):
    parameters = parseParameter(argv)

    train_file  = parameters["train_file"]
    model_file  = parameters["model_file"]
    sample_type = parameters["sample_type"];
    mem         = parameters["mem"]

    # read a instance to know the number of features and labels
    train_reader = ArffReader(train_file, 1)
    x, y, has_next = train_reader.read()
    parameters["num_feature"] = len(x[0])
    parameters["num_label"]   = len(y[0])
    train_reader.close()

    #train   
    sample = sampler.get_sample(parameters);
    if 1 == mem:
        model = train_mem(train_file, parameters, sample) 
    else:
        model = train(train_file, parameters, sample)    

    #write the model
    s = pickle.dumps(model)
    f = open(model_file, "w")
    f.write(s)
    f.close()

if __name__ == "__main__":
    main(sys.argv)
