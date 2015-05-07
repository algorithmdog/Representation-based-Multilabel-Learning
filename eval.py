#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/utils/Python_Utils")
sys.path.append(path + "/../utils/Python_Utils")

from latent_factor import *
from Float_Utils   import *
import logging, Logger
import numpy as np
import arffio
import pickle

def printUsages():
    print "Usage: eval.py result_file true_file"

def parseParameter(argv):
    if len(argv) < 3: #at least 3 paramters: eval.py result_file true_file"
        printUsages()
        exit(1)

    parameters = dict()
    parameters["result_file"]  = argv[len(argv) - 2]
    parameters["true_file"]    = argv[len(argv) - 1]
    return parameters


def hamming(p, t):
    if p.shape != t.shape:
        pi,pj = p.shape
        ti,tj = t.shape
        logger = logging.getLogger(Logger.project_name)
        logger.error("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))
        raise Exception("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))

    correct = 0
    total   = 0
    m,n = p.shape
    for i in xrange(m):
        for j in xrange(n):
            total   += 1
            if eq(p[i,j], t[i,j]):
                correct += 1
     
    return 1 - correct * 1.0 / total


def instance_F(p,t):
    if p.shape != t.shape:
        pi,pj = p.shape
        ti,tj = t.shape
        logger = logging.getLogger(Logger.project_name)
        logger.error("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))
        raise Exception("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))

    m,n = p.shape
    psum = np.sum(p,1)
    tsum = np.sum(t,1)
    F = 0.0
    for i in xrange(m):
        correct = 0
        for j in xrange(n):
            if eq(p[i,j],1) and eq(t[i,j],1):
                correct += 1
            
        pre = 0.0
        if eq(psum[i],0):
            pre = 0.0
        else:   
            pre = correct * 1.0 / psum[i]
        
        rec = 0.0
        if eq(tsum[i],0):
            rec =  0.0
        else:
            rec = correct * 1.0 / tsum[i]   

        if eq(pre,0) and eq(rec,0):
            F += 0
        else:
            F += 2*pre*rec/(pre+rec)
    
    return F/m


def label_F(p,t):
    if p.shape != t.shape:
        pi,pj = p.shape
        ti,tj = t.shape
        logger = logging.getLogger(Logger.project_name)
        logger.error("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))
        raise Exception("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))

    m,n = p.shape
    psum = np.sum(p,0)
    tsum = np.sum(t,0)
    F = 0.0

    for j in xrange(n):
        correct = 0
        for i in xrange(m):
            if eq(p[i,j],1) and eq(t[i,j],1):
                correct += 1

        pre = 0.0
        if eq(psum[j],0):
            pre = 0.0
        else:
            pre = correct * 1.0 / psum[j]

        rec = 0.0
        if eq(tsum[j],0):
            rec =  0.0
        else:
            rec = correct * 1.0 / tsum[j]

        if eq(pre,0) and eq(rec,0):
            F += 0
        else:
            F += 2*pre*rec/(pre+rec)

    return F/n

if __name__ == "__main__":
    parameters = parseParameter(sys.argv)

    result_file  = parameters["result_file"]
    true_file    = parameters["true_file"]

    reader         = arffio.ArffReader(result_file, batch = 1000000000000)
    _, p, has_next = reader.read()
    reader         = arffio.ArffReader(true_file, batch = 1000000000000)
    _, t, has_next = reader.read()

    ham = hamming(p, t)
    print "hamming loss:%f|"%ham,

    ins_f = instance_F(p,t)
    print "ins_f:%f|"%ins_f,

    label_f = label_F(p,t)
    print "label_f:%f|"%label_f
