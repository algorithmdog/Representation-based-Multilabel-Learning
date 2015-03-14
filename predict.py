#!/bin/python
import sys;
import os;
path  = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/utils/Python_Utils");
sys.path.append(path + "/../utils/Python_Utils");

from latent_factor import *;
import Logger;
import numpy as np;
import arffreader
import pickle

def printUsages():
    print "Usage: predict.py test_file model_file";

def parseParameter(argv):
    if len(argv) < 3: #at least 3 paramters: predict.py x_file model_file
        printUsages();
        exit(1);

    parameters = dict();
    parameters["test_file"]  = argv[len(argv) - 2];
    parameters["model_file"] = argv[len(argv) - 1];
    return parameters;


def predict(model, x):
    p     = model.ff(x);
    return p;


if __name__ == "__main__":
    parameters = parseParameter(sys.argv);

    test_file  = parameters["test_file"];
    model_file = parameters["model_file"];

    reader         = arffreader.ArffReader(test_file, batch = 1000000000000);
    x,y,has_next   = reader.read();

    #load model 
    f     = open(model_file, "r");
    s     = f.read();
    model = pickle.loads(s);
    p     = predict(model, x);

    print p;
