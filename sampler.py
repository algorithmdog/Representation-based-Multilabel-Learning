#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/utils/Python_Utils")
sys.path.append(path + "/../utils/Python_Utils")

from common import *
from arffio import *
import logging, Logger
import pickle
import numpy as np
import scipy.sparse as sp
import random
import math



class Sampler:
    def __init__(self, paramters):
        nocode = 0
    def update(self, y):
        nocode = 0
    def sample(self, y):
        nocode = 0

class FullSampler(Sampler):
    def sample(self, y):
        return None
 

class InstanceSampler(Sampler):
    def __init__(self, parameters):
        self.ratio = 5
        if "sr" in parameters:
            self.ratio = parameters["sr"]

    def sample(self, y):
        #sample = np.int_(y)  
        #sample = sp.lil_matrix(y)  
        #sample = y.copy()
        m,n  = y.shape  
        #num = np.sum(sample,1)
        #num = sparse_sum(sample,1)
        num  = np.asarray(y.sum(1))[:,0]
        num.astype(np.int32)
        num *= self.ratio
        #for i in xrange(len(num)):
        #    num[i] = self.ratio * int(num[i])
            #num[i] =  max(self.ratio * int(num[i]), int(n * 0.1))
        
        nonzero = y.nonzero()

        total = np.sum(num)
        cols  = np.random.random(total)
        #print "total",total
        cols *= n;
        cols  = cols.astype(np.int32).tolist()
        #print len(cols)
        cols += nonzero[1].tolist()
        #print len(cols)
        rows  = np.zeros(total)
        pre   = 0
        for i in xrange(m):
            after = pre + num[i]
            rows[pre:after] = i
            pre   = after
        rows  = rows.astype(np.int32).tolist()
        rows += nonzero[0].tolist()
        vals  = np.ones(len(rows)).tolist()
        
        sample = sp.csr_matrix((vals,(rows,cols)),(m,n))
        #for i in xrange(m):
        #    for j in xrange(min(int(num[i]), int(n/2))):
        #        idx = int(random.random() * n)
        #        if n == idx: idx = n - 1
        #        sample[i, idx] = 1
        #print len(sample.nonzero()[0])
        return sample



def get_sampler(parameters):

    if "st" not in parameters:
        logger = logging.getLogger(Logger.project_name)
        logger.error("Not sample_type provided by params in "
                     "sampler.get_sample")
        raise Exception("Not sample_type provided by params in"
                        " sampler.get_sample")
    
    sample_type = parameters["st"]
    
    if st.full_sampler == sample_type:
        return FullSampler(parameters)

    elif st.instance_sampler == sample_type:
        return InstanceSampler(parameters)

    else:
        logger = logging.getLogger(Logger.project_name)
        logger.error("Unknown sample_type %s"%sample_type)
        raise Exception("Unknown sample_type %s"%sample_type)


'''
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



