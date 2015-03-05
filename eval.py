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
    print "Usage: eval.py predict_file true_file";

def parseParameter(argv):
    if len(argv) < 3: #at least 3 paramters: eval.py predict_file true_file
        printUsages();
        exit(1);

    parameters = dict();
    parameters["pred_file"]   = argv[len(argv) - 2];
    parameters["true_file"]   = argv[len(argv) - 1];
    return parameters;

def eval(p, t):
    return hamming;


if __name__ == "__main__":
    parameters = parseParameter(sys.argv);
    p_file     = parameters["pred_file"];
    t_file     = parameters["true_file"];
    
    p_reader   = MatrixReader(p_file, batch = 10000000000);
    p          = p_reader.read();
    t_reader   = MatrixReader(t_file, batch = 10000000000);     
    t          = t_reader.read();

    hamming = eval(p, t);
