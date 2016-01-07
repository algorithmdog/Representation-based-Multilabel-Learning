#!/bin/python
import sys;
import os;

from common        import *
from latent_factor import *;
import logging, Logger;
import arffio;
import pickle;
import numpy as np;
import copy;
import time;

def printUsages():
    print "Usage: predict.py test_file result_file model_file";

def parseParameter(argv):
    if len(argv) < 4: #at least 3 paramters: predict.py test_file result_file model_file
        printUsages();
        exit(1);

    parameters = dict();
    parameters["test_file"]   = argv[len(argv) - 3];
    parameters["result_file"] = argv[len(argv) - 2]; 
    parameters["model_file"]  = argv[len(argv) - 1];
    return parameters;


def predict(model, x):
    p  = model.ff(x);
    
    m,n = p.shape;
    '''    
    for i in xrange(m):
        for j in xrange(n):
            if p[i,j] > model.thrsel.threshold:
                p[i,j] = 1;
            else:
                p[i,j] = 0;
    
    '''
    p[p > model.thrsel.threshold] = 1
    p[p != 1 ] = 0
    return p;


if __name__ == "__main__":
    logger = logging.getLogger(Logger.project_name)
    
    parameters = parseParameter(sys.argv);
    test_file   = parameters["test_file"];
    model_file  = parameters["model_file"];
    result_file = parameters["result_file"];

    reader  = arffio.SvmReader(test_file, batch = 1000000000000);
    x, _    = reader.full_read();

    model = Model(dict())
    model.load(model_file)


    #import cProfile, pstats, StringIO
    #pr =  cProfile.Profile()
    #pr.enable()
    
    start = time.time()
    p     = predict(model, x);
    end   = time.time()
    logger.info("predict time is %f seconds"%((end-start)))        
    
    #pr.disable()
    #s = StringIO.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
    #ps.print_stats()
    #print "update",s.getvalue()

    ##predctions to sparse data
    y = sp.csr_matrix(p)
    x = np.zeros((p.shape[0],1))
    for i in xrange(p.shape[0]):
        x[i][0] = 1
    x = sp.csr_matrix(x)

    #write
    writer = arffio.SvmWriter(result_file, model.num_feature, model.num_label);
    writer.write(x,y);
    writer.close();
     
        
