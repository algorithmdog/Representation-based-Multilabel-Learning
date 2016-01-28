#!/bin/python
import sys
import os


import numpy as np
import arffio
import pickle

def printUsages():
    print "Usage: imR.py data_file"

def parseParameter(argv):
    if len(argv) < 2: #at least 3 paramters: eval.py result_file true_file"
        printUsages()
        exit(1)

    parameters = dict()
    parameters["data_file"]  = argv[len(argv) - 1]
    return parameters

 
def imRcompute(y):

    m,n        = y.shape
    presents   = [0 for i in xrange(n)]
    
    pxys = dict()
    ij  = y.nonzero()
    for k in xrange(len(ij[0])):
        i = ij[0][k]
        j = ij[1][k]
        presents[j] += 1


    imR = 0
    num = n * 1.0
    for j in xrange(n):
        if 0 == presents[j]:
            num -= 1
        else:
            imR += (n-presents[j]) / presents[j]
  

    return imR/num

if __name__ == "__main__":
    parameters = parseParameter(sys.argv)

    data_file  = parameters["data_file"]
    reader     = arffio.SvmReader(data_file, batch = 1000000000000)
    _, y       = reader.full_read()


    imR        = imRcompute(y)
    print "imR of %s is %f"%(data_file,  imR)
