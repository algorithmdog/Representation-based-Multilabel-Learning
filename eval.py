#!/bin/python
import sys;
import os;
path  = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/utils/Python_Utils");
sys.path.append(path + "/../utils/Python_Utils");

from latent_factor import *;
from Float_Utils   import *;
import Logger;
import numpy as np;
import arffio;
import pickle;

def printUsages():
    print "Usage: eval.py result_file true_file";

def parseParameter(argv):
    if len(argv) < 3: #at least 3 paramters: eval.py result_file true_file"
        printUsages();
        exit(1);

    parameters = dict();
    parameters["result_file"]  = argv[len(argv) - 2];
    parameters["true_file"]    = argv[len(argv) - 1];
    return parameters;


def hamming(p, t):
    if p.shape != t.shape:
        pi,pj = p.shape;
        ti,tj = t.shape;
        raise Exception("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj));
        Logger.instance.error("p.shape(%d,%d) != t.shape(%d,%d)"%(pi,pj,ti,tj));

    correct = 0;
    total   = 0;
    m,n = p.shape;
    for i in xrange(m):
        for j in xrange(n):
            total   += 1;
            if eq(p[i,j], t[i,j]):
                correct += 1;
     
    return 1 - correct * 1.0 / total;


if __name__ == "__main__":
    parameters = parseParameter(sys.argv);

    result_file  = parameters["result_file"];
    true_file    = parameters["true_file"];

    reader         = arffio.ArffReader(result_file, batch = 1000000000000);
    _, p, has_next = reader.read();
    reader         = arffio.ArffReader(true_file, batch = 1000000000000);
    _, t, has_next = reader.read();

    ham = hamming(p, t);
    print "hamming loss:%f"%ham;
