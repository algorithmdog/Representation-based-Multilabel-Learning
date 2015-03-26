#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/utils/Python_Utils")
sys.path.append(path + "/../utils/Python_Utils")

from arffio import *
import logging, Logger
import pickle
import numpy as np
import random
import math
random.seed(0)

class CorrelationSampler:
    def __init__(self, parameters):
        self.parameters = dict()
        self.parameters["num_label"]  = 1000
        self.parameters["num_sample_factor"] = 20
        self.parameters["lambda"]     = 0.001
        
        if "num_label" in parameters:
            self.parameters["num_label"] = parameters["num_label"]
        if "num_sample_factor" in parameters:
            self.parameters["num_sample_factor"] = \
                parameters["num_sample_factor"]
        if "lambda" in parameters:
            self.parameters["lambda"] = parameters["lambda"]
            
        num_label  = self.parameters["num_label"]
        num_sample_factor = self.parameters["num_sample_factor"] 
        fan_in  = num_label
        fan_out = num_sample_factor

        r = math.sqrt( 6.0/(fan_in+fan_out) )
        w = [ [random.random() * 2 * r - r for j in xrange(num_label)]\
                                           for i in xrange(num_sample_factor) ]
        b = [ random.random() * 2 * r -r for j in xrange(num_label) ]
   
        self.w = np.array(w)
        self.b = np.array(b) 
     
    def update(self, y):
        lili = 0
    def predict(self, y):
        lili = 0
    def sample(self, y):
        lili = 0        

class InstanceSampler:
    def __init__(self, parameters):
        no_execute = 0

    def sample(self, y):
        sample = np.int_(y)         
        m,n = sample.shape  
        num = np.sum(sample,1)

        for i in xrange(m):
            for j in xrange(min(num[i], int(n/2))):

                idx = int(random.random() * n)
                if n == idx: idx = n - 1
                while 1 == sample[i][idx]:
                    idx = int(random.random() * n)
                    if n == idx: idx = n - 1
                sample[i, idx] = 1;
    
                                 
        return sample

class LabelSampler:
    def __init__(self, parameters):
        if "num_label" not in parameters:
            logger = logging.getLogger(Logger.project_name);
            logger.error("no num_label provided by paramters in label_sampler.init");           

        num_label = parameters["num_label"];
        self.num_ins    = 0;
        self.num_labels = np.array([0 for i in xrange(num_label)]);
    
    def update(self, y):
        m,n = y.shape
        self.num_ins    += m
        self.num_labels += np.sum(y, 0)    

    def sample(self, y):
        sample = np.copy(y)    
        return sample


def get_sampler(sample_type, parameters):
    
    if "full" == sample_type:
        return None;

    elif "instance_sample" == sample_type:
        return InstanceSampler(parameters);

    elif "label_sample" == sample_type:
        return LabelSampler(parameters);

    elif "correlation_sample" == sample_type:
        return CorrelationSampler(parameters);

    else:
        logger = logging.getLogger(Logger.project_name);
        logger.error("Unknown sample_type %s"%sample_type);
        raise Exception("Unknown sample_type %s"%sample_type);


backup = '''
def printUsages():
    print "Usage: sample.py [options] origin_file sample_file"

def parseParameter(argv):
    if len(argv) < 3: #at least 4 paramters: train.py train_file m_file
        printUsages()
        exit(1)

    parameters = dict()
    parameters["origin_file"]  = argv[len(argv) - 2]
    parameters["sample_file"]  = argv[len(argv) - 1]

    return parameters



def sample(parameters):
    origin_file = parameters["origin_file"]
    target_file = parameters["target_file"]    


if __name__ == "__main__":
    parameters = parseParameter(sys.argv)
'''


