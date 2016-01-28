#!/bin/python
import sys
import os

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
from common import *

np.random.seed(0)
random.seed(0)


def checkParamValid(param):
    if param["op"] == op.alternative_least_square:
        if act.linear != param["ha"] or act.linear != param["oa"]:
            print "alternative_least_square optimization requires linear "\
                  "hidden_activation and output_activation"
            return False;
        if  m.internal_memory != param["m"]:
            print "alternative_least_square optimization requires not to use external_memory"
            return False;

    if lo.negative_log_likelihood == param["l"] and act.sgmoid != param["oa"]:
            print "negative_log_likelihood loss requires sgmoid output_activation"      
            return False;
    if lo.least_square == param["l"] and act.linear != param["oa"]:
            print "least_square loss requires linear output_activation"
            return False;

    return True;


def train_internal(model, train_file, parameters):
    batch  = parameters["b"]
    niter  = parameters["i"]
    sample = sampler.get_sampler(parameters)   
    logger = logging.getLogger(Logger.project_name)
    logger.info("Model initialization done")

    train_reader = SvmReader(train_file)
    x, y   = train_reader.full_read() 
    num, _ = y.shape
    logger.info("Training data loading done")

    sample.update(y)
    logger.info("Sampling initialization done")
    start_time = time.time()

    if op.gradient == parameters["op"]:
        for iter1 in xrange(niter):
            start = 0
            end = batch
            while start < num:
                #logger.info("start = %d, end = %d\n"%(start, end))
                if end > num:   end = num
            
#               import cProfile, pstats, StringIO
#               pr =  cProfile.Profile()
#               pr.enable()

                batch_x = x[start:end, :]
                batch_y = y[start:end, :] 
                batch_i = sample.sample(batch_y)
                model.grad_update(batch_x, batch_y, batch_i)      

                start += batch;
                end += batch;
#               pr.disable()
#               s = StringIO.StringIO()
#               sortby = 'cumulative'
#               ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
#               ps.print_stats()
#               print "update",s.getvalue()
            
            logger.info("The %d-th iteration completes"%(iter1+1)); 
    elif op.alternative_least_square == parameters["op"]:
        model.b[0] = np.zeros(model.b[0].shape)
        model.lb   = np.zeros(model.lb.shape)

        for iter1 in xrange(niter):
            model.al_update(x, y, None)
            logger.info("The %d-th iteration completes"%(iter1+1));

    else:
        logger.error("Invalid optimization scheme (%s)"%paramters["op"])

    #####tuning the threshold
    total = 0
    start = 0
    end = batch
    while start < num and total < 1000:
        if end > num: end = num
        batch_x = x[start:end,:]
        batch_y = y[start:end,:]
        batch_p = model.ff(batch_x)
        model.thrsel.update(batch_p, batch_y)
        start += batch
        end   += batch
        total += 1

    logger.info("The threshold tuning completes") 
    end_time = time.time()
    logger.info("The training time is %f"%(end_time-start_time))

    return model


def train_external(model, train_file, parameters):

    batch  = parameters["b"]
    niter  = parameters["i"]
    sample = sampler.get_sampler(parameters)
    logger = logging.getLogger(Logger.project_name)


    ##initilization the sampler
    train_reader   = SvmReader(train_file, batch)
    has_next       = True
    while has_next:
        x,y,has_next = train_reader.read()
        sample.update(y)
    
    ##weight updates
    for iter1 in xrange(niter): 
        train_reader = SvmReader(train_file, batch)
    
        has_next = True
        while has_next:
            x, y, has_next = train_reader.read()
            idx            = sample.sample(y)
            model.grad_update(x, y, idx)

        logger.info("The %d-th iteration completes"%(iter1+1)); 
        train_reader.close()

    ##tuning threshold
    train_reader = SvmReader(train_file, batch)
    x, y, has_next = train_reader.read()
    while has_next:
        p = model.ff(x)
        model.thrsel.update(p, y)
        x, y, has_next = train_reader.read()
        

    return model

