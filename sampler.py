#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/utils/Python_Utils")
sys.path.append(path + "/../utils/Python_Utils")

from arffio import *
from util   import *
import logging, Logger
import pickle
import numpy as np
import scipy.sparse as sp
import random
import math
import Roulette


class Sampler:
    def __init__(self, paramters):
        nocode = 0
    def update(self, y):
        nocode = 0
    def sample(self, y):
        return sp.lil_matrix(y.shape)

class FullSampler(Sampler):
    def sample(self, y):
        return sp.lil_matrix(y.shape)

class CorrelationSampler(Sampler):
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

class InstanceSampler(Sampler):
    def __init__(self, parameters):
        self.ratio = 1
        if "sample_ratio" in parameters:
            self.ratio = parameters["sample_ratio"]
        print "self.ratio",self.ratio

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



class NegativeSampler(Sampler):
    def __init__(self, parameters):
        self.distribution = None
        self.num = 0
        self.ratio = 5
        if "sample_ratio" in parameters:
            self.ratio = parameters["sample_ratio"]

    def update(self, y):
        m,n = y.shape
        if self.distribution is None:
            self.distribution = np.array([0.0 for i in xrange(n)])
            self.num          = 0

        self.distribution  *= self.num

        xy = y.nonzero()
        for k in xrange(len(xy[0])):
            i = xy[0][k]
            j = xy[1][k]
            self.distribution[j] += 1
        self.num += len(xy[0])

        self.distribution  /= self.num

    
    def sample(self, y):
        
        sample = sp.lil_matrix(y)
        m,n = sample.shape
        num = sparse_sum(sample, 1)
        for i in xrange(len(num)):
            num[i] = self.ratio * int(num[i])

        for i in xrange(m):
            samplenum = min(num[i], n - num[i])
            
            if samplenum == n - num[i]:
                for j in xrange(n):
                    sample[i,j] = 1
            else:
                roulette = '''
                for i in xrange(samplenum):
                    idx = Roulette.roulette_pick(self.distribution)
                    if n == idx: idx = n - 1
                    while 1 == sample[i, idx]:
                        idx = Roulette.roulette_pick(self.distribution)
                        if n == idx: idx = n - 1

                    sample[i, idx] = 1
                '''
                for j in xrange(n):
                    r = random.random()
                    if r < ((num[i] /self.ratio) * (self.ratio +1)):
                        sample[i,j] = 1 
   

        return sample



def get_sample(parameters):

    if "sample_type" not in parameters:
        logger = logging.getLogger(Logger.project_name)
        logger.error("Not sample_type provided by params in "
                     "sampler.get_sample")
        raise Exception("Not sample_type provided by params in"
                        " sampler.get_sample")
    
    sample_type = parameters["sample_type"]
    
    if "full" == sample_type:
        return FullSampler(parameters)

    elif "instance_sample" == sample_type:
        return InstanceSampler(parameters)

    elif "correlation_sample" == sample_type:
        return CorrelationSampler(parameters)

    elif "negative_sample" == sample_type:
        return NegativeSampler(parameters)

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



