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
import sample

def printUsages():
    print "Usage: train.py [options] train_file m_file"
    print "options"
    print "   -i: ins lambda, the instance regularization coefficient (default 0.001)"
    print "   -l: label lambda, the label regularization coefficient (default 0.001)" 
    print "   -s: sizes, the architecture: [num_node_layer1,num_node_layer2,...] (default [])"
    print "   -b: batch, the number of instances in a batch (default 100)"
    print "   -n: num of iter, the number of iterations (default 20)"
    print "   -f: idx file, the arff file containing sampling result (if not specified, full)"

def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file m_file
        printUsages()
        exit(1)

    parameters = dict()
    parameters["train_file"]   = argv[len(argv) - 2]
    parameters["m_file"]       = argv[len(argv) - 1]
    parameters["ins_lambda"]   = 0.001
    parameters["label_lambda"] = 0.001
    parameters["sizes"]        = [] 
    parameters["batch"]        = 100
    parameters["niter"]        = 20
    parameters["idx_file"]     = None # When idx_file = None, full
   
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
        elif "-f" == argv[i]:
            parameters["idx_file"] = argv[i+1]
        else:
            printUsages()
            exit(1)
        i += 2

    return parameters


def train(train_file, idx_file, parameters):
    model = Model(parameters)
    
    
    batch = parameters["batch"]
    niter = parameters["niter"]

    for iter1 in xrange(niter): 
        train_reader = ArffReader(train_file, batch)

        #idx_file specified
        if None != idx_file:
            idx_reader   = ArffReader(idx_file,batch)
       
            x, y,has_next = train_reader.read()
            _, idx        = idx_reader.read()
            while has_next:
                model.update(x, y, idx)
                x, y, has_next = train_reader.read()
                _, idx         = idx_reader.read()

            idx_reader.close()
        #idx_file not specified, full
        else:
            x, y, has_next = train_reader.read()
            idx            = np.ones(y.shape)

            while has_next:
                model.update(x, y, idx)    
                x, y, has_next = train_reader.read()
                idx = np.ones(y.shape)   

       
        train_reader.close()

    return model


if __name__ == "__main__":
    parameters = parseParameter(sys.argv)

    train_file = parameters["train_file"]
    idx_file   = parameters["idx_file"]
    m_file     = parameters["m_file"]
    
    # read a instance to know the number of features and labels
    train_reader = ArffReader(train_file, 1)
    x, y, has_next = train_reader.read()
    parameters["num_feature"] = len(x[0])
    parameters["num_label"]   = len(y[0])
    train_reader.close()

    #train   
    model = train(train_file, idx_file, parameters) 
    
    #write the model
    s = pickle.dumps(model)
    f = open(m_file, "w")
    f.write(s)
    f.close()
