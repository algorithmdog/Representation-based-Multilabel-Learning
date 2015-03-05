#!/bin/python
import sys;
import os;
path  = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/utils/Python_Utils");
sys.path.append(path + "../utils/Python_Utils");

from latent_factor import *;
from io            import *;
import Logger;
import numpy as np;

def printUsages():
    print "Usage: train.py [options] x_file y_file m_file";
    print "options";
    print "-ins_lambda, the instance regularization coefficient (default 0.001)";
    print "-label_lambda, the label regularization coefficient (default 0.001)"; 
    print "-sizes, the architecture: [num_node_layer1,num_node_layer2,...] (default [])";
    print "-batch, the number of instances in a batch (default 100)";
    print "-niter, the number of iterations (default 20)";

def parseParameter(argv):
    if len(argv) < 4: #at least 4 paramters: train.py x_file y_file m_file
        printUsages();
        exit(1);

    parameters = dict();
    parameters["x_file"]       = argv[len(argv) - 3];
    parameters["y_file"]       = argv[len(argv) - 2];
    parameters["m_file"]       = argv[len(argv) - 1];
    parameters["ins_lambda"]   = 0.001;
    parameters["label_lambda"] = 0.001;
    parameters["sizes"]        = []; 
    parameters["batch"]        = 100;
    parameters["niter"]        = 20;
   
    i = 1;
    while i + 1 < len(argv) - 3:
        if  "-ins_lambda" == argv[i]:
            parameters["ins_lambda"]   = float(argv[i+1]);
        elif "-label_lambda" == argv[i]:
            parameters["label_lambda"] = float(argv[i+1]);
        elif "-sizes" == argv[i]:
            line  = argv[i+1];
            line  = line.strip();
            line  = line[1:len(line)-1];
            line  = line.strip();
            if "" != line:
                eles  = line.split(",");
                sizes = map(int, eles);
                parameters["sizes"] =  sizes;
        elif "-batch"  == argv[i]:
            parameters["batch"] = int(argv[i+1]); 
        elif "-niter" == argv[i]:
            parameters["niter"] = int(argv[i+1]);   
        else:
            printUsages();
            exit(1);
        i += 2;

    return parameters;


def train(x, y, idx, parameters):
    model = Model(parameters);

    batch = parameters["batch"];
    niter = parameters["niter"];

    for iter1 in niter:
        start = 0;
        end   = batch;
        while end <= num_instance:
            if end > num_instance:  end = num_instance;
            batch_x   = ...;
            batch_y   = ...;
            batch_idx = ...;
            model.update(batch_x, batch_y, batch_idx);
            start += batch;
            end   += batch;
    return model;


if __name__ == "__main__":
    parameters = parseParameter(sys.argv);

    x_file     = parameters["x_file"];
    y_file     = parameters["y_file"];
    m_file     = parameters["m_file"];

    x_reader   = io.MatrixReader(x_file, batch = 1000000000000);
    x          = x_reader.read();
    y_reader   = io.MatrixReader(y_file, batch = 1000000000000);
    y          = y_reader.read();
    idx        = np.copy(y); #build   
 
    num_feature  = ...
    num_label    = ...
    num_instance = ..    
    parameters["num_feature"] = num_feature;
    parameters["num_label"]   = num_label;
   
    model = train(x, y, idx, parameters); 
    model.write(m_file);
