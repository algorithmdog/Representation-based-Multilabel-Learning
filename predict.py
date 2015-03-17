#!/bin/python
import sys;
import os;
path  = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/utils/Python_Utils");
sys.path.append(path + "/../utils/Python_Utils");

from latent_factor import *;
import Logger;
import arffio;
import pickle;
import numpy as np;
import copy;

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
    for i in xrange(m):
        for j in xrange(n):
            if p[i,j] > 0.5:
                p[i,j] = 1;
            else:
                p[i,j] = 0;
    return p;


if __name__ == "__main__":
    parameters = parseParameter(sys.argv);

    test_file   = parameters["test_file"];
    model_file  = parameters["model_file"];
    result_file = parameters["result_file"];

    reader           = arffio.ArffReader(test_file, batch = 1000000000000);
    x, y, has_next   = reader.read();
    
    #load model and predition
    f     = open(model_file, "r");
    s     = f.read();
    model = pickle.loads(s);
    p     = predict(model, x);
    

    ##predctions to obj
    obj         = copy.deepcopy(reader.nextobj);

    obj["attributes"] = [];
    for j in xrange(len(reader.nextobj["attributes"])):
        if arffio.flag_label in reader.nextobj["attributes"][j][0]: 
            obj["attributes"].append( reader.nextobj["attributes"][j] );
    
    obj["data"] = [ [0 for col in xrange(len(p[0]))] for row in xrange(len(p))  ];      
    for i in xrange(len(p)):
        for j in xrange(len(p[0])):
            obj["data"][i][j] = p[i][j];

    #write
    writer = arffio.ArffWriter(result_file);
    writer.write(obj);
    writer.close();
     
        
