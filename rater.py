#!/bin/python
import os
import sys

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
sys.path.append(path + "/../utils/Python_Utils")
sys.path.append(path + "/../Python_Utils")

from active       import *
from Matrix_Utils import *
from threshold    import *
import numpy as np
import scipy.sparse as sp
import util
import math
import pickle
import random
import logging, Logger
import types

class LearnRate:
    def __init__(self, model):
        ##the w and b for instances
        learnrate = float(model.parameters["learnrate"])
        self.rate_b = []
        self.rate_w = []
        for idx in xrange(len(model.num)-1):
            #rate_w = [ [learnrate for j in xrange(model.num[idx+1])] \
            #                      for i in xrange(model.num[idx]) ]
            #rate_b = [ learnrate for j in xrange(model.num[idx+1]) ]
            #self.rate_w.append(np.array(rate_w))
            #self.rate_b.append(np.array(rate_b))
            rate_w = learnrate
            rate_b = learnrate
            self.rate_w.append(rate_w)
            self.rate_b.append(rate_b)

        ## the lw and lb for labels
        #self.rate_lb  = np.array([ learnrate for j in xrange(model.num_label) ])
        #self.rate_lw  = np.array([ [learnrate for j in xrange(model.num_label)] \
        #                                      for i in xrange(model.num_factor) ] )
        self.rate_lb = learnrate
        self.rate_lw = learnrate
    
    def compute_rate(self, model):
        nocommand = 0
    def update_before_paramupdate(self, model):
        nocommand = 0    
    def update_after_paramupdate(self, model):
        nocommand = 0

##################The AdaGrad #################
class AdaGrad(LearnRate):
    def __init__(self, model):
        LearnRate.__init__(self, model)        
        #set the initial_rate
        self.initial_rate = float(model.parameters["learnrate"])

        ##the w and b for instances
        self.ada_b = []
        self.ada_w = []
        for idx in xrange(len(model.num)-1):
            #ada_w = [ [ 1 for j in xrange(model.num[idx+1])] \
            #              for i in xrange(model.num[idx]) ]
            #ada_b = [ 1 for j in xrange(model.num[idx+1]) ]
            #self.ada_w.append(np.array(ada_w))
            #self.ada_b.append(np.array(ada_b))
            ada_w = np.ones((model.num[idx],model.num[idx+1]))
            ada_b = np.ones((model.num[idx+1]))
            self.ada_w.append(ada_w)
            self.ada_b.append(ada_b)

        ## the lw and lb for labels
        #self.ada_lb       = np.array([ 1 for j in xrange(model.num_label) ])
        #self.ada_lw       = np.array([ [ 1 for j in xrange(model.num_label)] \
        #                                   for i in xrange(model.num_factor) ] )
        self.ada_lb = np.ones((model.num_label))
        self.ada_lw = np.ones((model.num_factor, model.num_label));

    def update_before_paramupdate(self, model):
        #update the rms
        for i in xrange(len(model.grad_w)):
            grad_w = model.grad_w[i] + 2 * model.ins_lambda * model.w[i]
            grad_b = model.grad_b[i]
            self.ada_w[i] = np.sqrt(self.ada_w[i] * self.ada_w[i] \
                                    + grad_w * grad_w) 
            self.ada_b[i] = np.sqrt(self.ada_b[i] * self.ada_b[i] \
                                    + grad_b * grad_b)            

        grad_lw = model.grad_lw + 2 * model.label_lambda * model.lw
        grad_lb = model.grad_lb
        self.ada_lw = np.sqrt(self.ada_lw * self.ada_lw + grad_lw * grad_lw)
        self.ada_lb = np.sqrt(self.ada_lb * self.ada_lb + grad_lb * grad_lb)  

    def compute_rate(self, model): 
        for i in xrange(len(model.grad_w)):
            self.rate_w[i] = self.initial_rate / self.ada_w[i]            
            self.rate_b[i] = self.initial_rate / self.ada_b[i]
        self.rate_lw = self.initial_rate / self.ada_lw
        self.rate_lb = self.initial_rate / self.ada_lb


################ AdaDelta ######################
class AdaDelta(AdaGrad):
    def __init__(self, model):
        AdaGrad.__init__(self, model)
        
        initial_rate = float(model.parameters["learnrate"])
        self.delta_b = []
        self.delta_w = []
        for idx in xrange(len(model.num)-1):
            #delta_w = [ [ initial_rate for j in xrange(model.num[idx+1])] \
            #                           for i in xrange(model.num[idx]) ]
            #delta_b = [ initial_rate for j in xrange(model.num[idx+1]) ]
            #self.delta_w.append(np.array(delta_w))
            #self.delta_b.append(np.array(delta_b))

            delta_w = np.zeros((model.num[idx],model.num[idx+1])) + initial_rate
            delta_b = np.zeros((model.num[idx],model.num[idx+1])) + initial_rate
            self.delta_w.append(delta_w)
            self.delta_b.append(delta_b)

        ## the lw and lb for labels
        #self.delta_lb = np.array([ initial_rate for j in xrange(model.num_label) ])
        #self.delta_lw = np.array([ [initial_rate for j in xrange(model.num_label)] \
        #                                         for i in xrange(model.num_factor) ] )
        self.delta_lb = np.zeros(model.num_label) + initial_rate
        self.delta_lw = np.zeros((model.num_factor,model.num_label)) + initial_rate
        
    def compute_rate(self, model):
        for i in xrange(len(model.grad_w)):
            self.rate_w[i] = self.delta_w[i] / self.ada_w[i]
            self.rate_b[i] = self.delta_b[i] / self.ada_b[i]
        self.rate_lw = self.delta_lw / self.ada_lw
        self.rate_lb = self.delta_lb / self.ada_lb
        
    def update_after_paramupdate(self, model):
        for i in xrange(len(model.grad_w)):
            delta_w = self.rate_w[i] * (model.grad_w[i] + 2 * model.ins_lambda * model.w[i])
            delta_b = self.rate_b[i] * model.grad_b[i]
            self.delta_w[i] = np.sqrt(self.delta_w[i] * self.delta_w[i] \
                                    + delta_w * delta_w)
            self.delta_b[i] = np.sqrt(self.delta_b[i] * self.delta_b[i] \
                                    + delta_b * delta_b)

        delta_lw = self.rate_lw * (model.grad_lw + 2 * model.label_lambda * model.lw)
        delta_lb = self.rate_lb * model.grad_lb
        self.delta_lw = np.sqrt(self.delta_lw * self.delta_lw + delta_lw * delta_lw)
        self.delta_lb = np.sqrt(self.delta_lb * self.delta_lb + delta_lb * delta_lb)        


 
        
