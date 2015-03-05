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
    print "Usage: predict.py x_file model_file";

def parseParameter(argv):
    if len(argv) < 3: #at least 3 paramters: predict.py x_file model_file
        printUsages();
        exit(1);

    parameters = dict();
    parameters["x_file"]   = argv[len(argv) - 2];
    parameters["m_file"]   = argv[len(argv) - 1];
    return parameters;


def predict(model, x):
    p     = model.ff(x);
    return p;


if __name__ == "__main__":
    parameters = parseParameter(sys.argv);

    x_file     = parameters["x_file"];
    m_file     = parameters["m_file"];

    x_reader   = io.MatrixReader(x_file, batch = 1000000000000);
    x          = x_reader.read();

    model = Model(parameters).read(m_file); 
    p     = predict(model, x);
