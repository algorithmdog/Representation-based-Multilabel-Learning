#!/bin/python
import os
import sys

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)

from common       import *
from active       import *
from threshold    import *
import numpy as np
import scipy.sparse as sp
import math
import pickle
import random
import logging, Logger
import types


class LearnRate:
    def __init__(self, model):
        ##the w and b for instances
        learnrate = float(model.learnrate)
        self.rate_b = []
        self.rate_w = []
        self.nonzero = dict()

        for idx in xrange(len(model.w)):
            rate_w = (model.w[idx] - model.w[idx]) + learnrate
            rate_b = (model.b[idx] - model.b[idx]) + learnrate
            self.rate_w.append(rate_w)
            self.rate_b.append(rate_b)
        
        self.rate_lb = (model.lb - model.lb) + learnrate
        self.rate_lw = (model.lw - model.lw) + learnrate
    
    def compute_rate(self, model):
        nocommand = 0

    def update_before_paramupdate(self, model):
        for i in xrange(len(model.grad_w)):
            if sp.isspmatrix(model.grad_w[i]):
                nonzero = model.grad_w[i].nonzero()
                self.nonzero["%d"%i] = nonzero

        #print type(model.grad_lw)        
        if sp.isspmatrix(model.grad_lw):
            nonzero = model.grad_lw.nonzero()
            #t1,t2 = model.grad_lw.shape
            #print "grad_lw sparity",len(nonzero[0])*1.0/t1 /t2
            self.nonzero['l'] = nonzero


    def update_after_paramupdate(self, model):
        nocommand = 0

##################The AdaGrad #################
class AdaGrad(LearnRate):
    def __init__(self, model):
        LearnRate.__init__(self, model)        
        #set the initial_rate
        self.initial_rate = float(model.learnrate)

        ##the w and b for instances
        self.ada_b = []
        self.ada_w = []
        for idx in xrange(len(model.w)):
            ada_w = 1 + (model.w[idx] - model.w[idx])
            ada_b = 1 + (model.b[idx] - model.b[idx])
            self.ada_w.append(ada_w)
            self.ada_b.append(ada_b)

        self.ada_lb  = np.ones((model.num_label))
        self.ada_lw  = np.ones((model.num_factor, model.num_label));
        self.nonzero = dict()


    def update_before_paramupdate(self, model):
        for i in xrange(len(model.grad_w)):
            if sp.isspmatrix(model.grad_w[i]):
                nonzero = model.grad_w[i].nonzero()
                #t1,t2 = model.grad_w[i].shape
                #print "grad_w sparity",len(nonzero[0]) * 1.0 / t1 /t2
                self.nonzero["%d"%i] = nonzero
                grad_w  = np.asarray(model.grad_w[i].data) + 2 * (model.l2_lambda) * model.w[i][nonzero]
                self.ada_w[i][nonzero] = self.ada_w[i][nonzero] + grad_w * grad_w            
            else:
                grad_w = model.grad_w[i] + 2 * (model.l2_lambda) * model.w[i]
                self.ada_w[i] = self.ada_w[i] + grad_w * grad_w
            
            grad_b = model.grad_b[i]
            self.ada_b[i] = self.ada_b[i] + grad_b * grad_b            


        #print type(model.grad_lw)        
        if sp.isspmatrix(model.grad_lw):
            nonzero = model.grad_lw.nonzero()
            #t1,t2 = model.grad_lw.shape
            #print "grad_lw sparity",len(nonzero[0])*1.0/t1 /t2
            self.nonzero['l'] = nonzero
            grad_lw = np.asarray(model.grad_lw.data) + 2 * model.l2_lambda * model.lw[nonzero]
            self.ada_lw[nonzero] = self.ada_lw[nonzero] + grad_lw * grad_lw      
        else:
            grad_lw = model.grad_lw + 2 * model.l2_lambda * model.lw
            self.ada_lw = self.ada_lw + grad_lw * grad_lw
        
        grad_lb = model.grad_lb
        self.ada_lb = self.ada_lb + grad_lb * grad_lb  


    def compute_rate(self, model): 
        for i in xrange(len(model.grad_w)):
            if sp.isspmatrix(model.grad_w[i]):
                nonzero = self.nonzero['%d'%i]
                self.rate_w[i][nonzero] = self.initial_rate / np.sqrt(self.ada_w[i][nonzero])
            else:
                self.rate_w[i] = self.initial_rate / np.sqrt(self.ada_w[i])            
            self.rate_b[i] = self.initial_rate / np.sqrt(self.ada_b[i])

        if sp.isspmatrix(model.grad_lw):
            nonzero = self.nonzero['l']
            self.rate_lw[nonzero] = self.initial_rate / np.sqrt(self.ada_lw[nonzero])
        else:
            self.rate_lw = self.initial_rate / np.sqrt(self.ada_lw)
        self.rate_lb = self.initial_rate / np.sqrt(self.ada_lb)


