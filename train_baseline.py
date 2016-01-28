#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]

from latent_factor import *
from arffio        import *
from common        import *
import copy
import logging, Logger
import pickle
import numpy as np
import scipy.sparse as sp
import sampler
import random
import time
from common       import *
from train_common import *

np.random.seed(0)
random.seed(0)

def printUsages():
    print "Usage: train_rep.py train_file model_file"


def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file model_file
        printUsages()
        exit(1)

    parameters = copy.deepcopy(leml_default_params)
    if False == checkParamValid(parameters):
        printUsages()
        exit(1)    

    parameters["train_file"]   = argv[len(argv) - 2]
    parameters["model_file"]   = argv[len(argv) - 1]

    return parameters


def main(argv):
    parameters = parseParameter(argv)
    
    train_file  = parameters["train_file"]
    model_file  = parameters["model_file"]

    # read a instance to know the number of features and labels
    train_reader       = SvmReader(train_file, 1)
    x, y, has_next     = train_reader.read()
    parameters["nx"]   = x.shape[1]
    parameters["ny"]   = y.shape[1]
    train_reader.close()

    model              = Model(parameters)
    thrsel             = ThresholdSel()
    thrsel.threshold   = 10000000.0
    model.thrsel       = thrsel
     
    #write the model
    #model.clear_for_save()
    model.save(model_file)
    #s = pickle.dumps(model)
    #f = open(model_file, "w")
    #f.write(s)
    #f.close()

if __name__ == "__main__":
    main(sys.argv)
