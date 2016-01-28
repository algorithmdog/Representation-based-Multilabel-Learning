#!/bin/python
import sys
import os
path  = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/utils/Python_Utils")
sys.path.append(path + "/../utils/Python_Utils")

from latent_factor import *
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
    parameters["result_file"]  = argv[len(argv) - 1]
    parameters["true_file"]    = argv[len(argv) - 2]
    return parameters


def check(p, t):
    if p.shape != t.shape:
        pi, pj = p.shape
        ti, tj = t.shape    
        logger = logging.getLogger(Logger.project_name)
        logger.error("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))
        raise Exception("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj))
        return False
    return True   
 



def acc(p, t):

    check(p, t)

    m,n   = p.shape
    total = m * n


    numerator = [0 for i in xrange(m)]
    dominator = [0 for i in xrange(m)]
    pxys      = dict()
    xy        = p.nonzero()
    for i in xrange(len(xy[0])):
        x = xy[0][i]
        y = xy[1][i]
        pxys["%d_%d"%(x,y)] = 1
        dominator[x]       += 1

    xy        = t.nonzero()
    for i in xrange(len(xy[0])):
        x = xy[0][i]
        y = xy[1][i]
        if "%d_%d"%(x,y) in pxys:
            numerator[x] += 1    
        else: 
            dominator[x] += 1

    acc = 0.0
    for i in xrange(m):
        if 0 == dominator[i]:
            acc += 0.0
        else:
            acc += 1.0 * numerator[i] / dominator[i]
    return acc / m 



def hamming(p, t):

    check(p, t)


    m,n   = p.shape
    total = m * n


    nump = 0
    pxys = dict()
    xy  = p.nonzero()
    for i in xrange(len(xy[0])):
        x = xy[0][i]
        y = xy[1][i]
        pxys["%d_%d"%(x,y)] = 1
        nump += 1

    correct = 0
    numt    = 0
    xy      = t.nonzero()
    for i in xrange(len(xy[0])):
        x = xy[0][i]
        y = xy[1][i]
        numt += 1
        if "%d_%d"%(x,y) in pxys:
            correct += 1    
 
    return (numt + nump - 2 * correct) * 1.0 / total


def instance_F(p,t):
    check(p,t)

    m,n = p.shape
    correct  = [0 for i in xrange(m)]
    lenp     = [0 for i in xrange(m)]
    lent     = [0 for i in xrange(m)]

    pxys = dict()
    xy  = p.nonzero()
    for i in xrange(len(xy[0])):
        x = xy[0][i]
        y = xy[1][i]
        lenp[x] += 1
        pxys["%d_%d"%(x,y)] = 1

    xy      = t.nonzero()
    for k in xrange(len(xy[0])):
        x = xy[0][k]
        y = xy[1][k]
        lent[x] += 1
        if "%d_%d"%(x,y) in pxys:
            correct[x] += 1

    F = 0
    for i in xrange(m):
        if lenp[i] !=0 or lent[i] != 0:
            F += 2.0 * correct[i]/(lenp[i] + lent[i])   
 
    return F/m


def label_F(p,t):
    check(p,t)

    m,n = p.shape
    correct  = [0 for i in xrange(n)]
    lenp     = [0 for i in xrange(n)]
    lent     = [0 for i in xrange(n)]
    
    pxys = dict()
    xy  = p.nonzero()
    for k in xrange(len(xy[0])):
        x = xy[0][k]
        y = xy[1][k]
        lenp[y] += 1
        pxys["%d_%d"%(x,y)] = 1

    xy      = t.nonzero()
    for k in xrange(len(xy[0])):
        x = xy[0][k]
        y = xy[1][k]
        lent[y] += 1
        if "%d_%d"%(x,y) in pxys:
            correct[y] += 1

    F = 0
    for j in xrange(n):
        if lenp[j] !=0 or lent[j] != 0:
            F += 2.0 * correct[j]/(lenp[j] + lent[j])

    return F/n

if __name__ == "__main__":
    parameters = parseParameter(sys.argv)

    result_file  = parameters["result_file"]
    true_file    = parameters["true_file"]

    reader   = arffio.SvmReader(result_file, batch = 1000000000000)
    _, p     = reader.full_read()
    reader   = arffio.SvmReader(true_file, batch = 1000000000000)
    _, t     = reader.full_read()


    ham = hamming(p, t)
    print "hamming loss:%f|"%ham,

    ins_f = instance_F(p,t)
    print "ins_f:%f|"%ins_f,

    label_f = label_F(p,t)
    print "label_f:%f|"%label_f,

    acc = acc(p, t)
    print "acc:%f|"%acc,


    print ""
