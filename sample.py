#!/bin/python
import sys;
import os;
path  = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/utils/Python_Utils");
sys.path.append(path + "/../utils/Python_Utils");

from arffio import *;
import logging, Logger;
import pickle;
import numpy as np;
import random;
random.seed(0);

class samplemodel:
    def __init__(self, parameters):
        self.parameters = dict();
        self.parameters["num_label"]  = 1000;
        self.parameters["num_sample_factor"] = 20;
        self.parameters["lambda"]     = 0.001;
        
        if "num_label" in parameters:
            self.parameters["num_label"] = parameters["num_label"];
        if "num_sample_factor" in parameters:
            self.parameters["num_sample_factor"] = parameters["num_sample_factor"];
        if "lambda" in parameters:
            self.parameters["lambda"] = parameters["lambda"];
            
        self.num_label  = self.parameters["num_label"];
        self.num_sample_factor = self.parameters["num_sample_factor"]; 

        fan_in = num_label;
        fan_in = num_sample_factor;
        r      = math.sqrt( 6.0/(fan_in+fan_out) );

        w = [ [random.random() * 2 * r - r for j in xrange(num_label)]\
                                           for i in xrange(num_sample_factor) ];
        b = [ random.random() * 2 * r -r for j in xrange(num_label) ];
   
        self.w = np.array(w);
        self.b = np.array(b); 
     
    def update(y):
    
    def predict(y):

    def sample(y):
        


def printUsages():
    print "Usage: sample.py [options] origin_file sample_file";

def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file m_file
        printUsages();
        exit(1);

    parameters = dict();
    parameters["origin_file"]  = argv[len(argv) - 2];
    parameters["sample_file"]  = argv[len(argv) - 1];

    return parameters;



def sample(parameters):
    origin_file = parameters["origin_file"];
    target_file = parameters["target_file"];    


if __name__ == "__main__":
    parameters = parseParameter(sys.argv);
    sample(parameters);
