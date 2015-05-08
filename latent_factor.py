#!/bin/python
import os
import sys

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
sys.path.append(path + "/../utils/Python_Utils")
sys.path.append(path + "/../Python_Utils")

from active       import *
from Matrix_Utils import *
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
            rate_w = [ [learnrate for j in xrange(model.num[idx+1])] \
                                  for i in xrange(model.num[idx]) ]
            rate_b = [ learnrate for j in xrange(model.num[idx+1]) ]
            self.rate_w.append(np.array(rate_w))
            self.rate_b.append(np.array(rate_b))


        ## the lw and lb for labels
        self.rate_lb  = np.array([ learnrate for j in xrange(model.num_label) ])
        self.rate_lw  = np.array([ [learnrate for j in xrange(model.num_label)] \
                                              for i in xrange(model.num_factor) ] )
        
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
            ada_w = [ [ 1 for j in xrange(model.num[idx+1])] \
                          for i in xrange(model.num[idx]) ]
            ada_b = [ 1 for j in xrange(model.num[idx+1]) ]
            self.ada_w.append(np.array(ada_w))
            self.ada_b.append(np.array(ada_b))


        ## the lw and lb for labels
        self.ada_lb       = np.array([ 1 for j in xrange(model.num_label) ])
        self.ada_lw       = np.array([ [ 1 for j in xrange(model.num_label)] \
                                           for i in xrange(model.num_factor) ] )

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
            delta_w = [ [ initial_rate for j in xrange(model.num[idx+1])] \
                                       for i in xrange(model.num[idx]) ]
            delta_b = [ initial_rate for j in xrange(model.num[idx+1]) ]
            self.delta_w.append(np.array(delta_w))
            self.delta_b.append(np.array(delta_b))


        ## the lw and lb for labels
        self.delta_lb = np.array([ initial_rate for j in xrange(model.num_label) ])
        self.delta_lw = np.array([ [initial_rate for j in xrange(model.num_label)] \
                                                 for i in xrange(model.num_factor) ] )

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
            self.ada_b[i] = np.sqrt(self.delta_b[i] * self.delta_b[i] \
                                    + delta_b * delta_b[i])

        delta_lw = self.rate_lw * (model.grad_lw + 2 * model.label_lambda * model.lw)
        delta_lb = self.rate_lb * model.grad_lb
        self.delta_lw = np.sqrt(self.delta_lw * self.delta_lw + delta_lw * delta_lw)
        self.delta_lb = np.sqrt(self.delta_lb * self.delta_lb + delta_lb * delta_lb)        


############## Model ##############
class Model:
    def __init__(self,  parameters):
        self.parameters = dict()
        self.parameters["num_feature"]   = 100
        self.parameters["num_factor"]    = 500
        self.parameters["num_label"]     = 1000
        self.parameters["sizes"]         = []
        self.parameters["hidden_active"] = "tanh"
        self.parameters["output_active"] = "sgmoid"
        self.parameters["loss"]          = "negative_log_likelihood"
        self.parameters["learnrate"]     = 0.1
        self.parameters["ins_lambda"]    = 0.001
        self.parameters["label_lambda"]  = 0.001
        self.parameters["mem"]           = 0
        
        if "num_feature" in parameters:
            self.parameters["num_feature"]   = parameters["num_feature"]
        if "num_factor"    in parameters:
            self.parameters["num_factor"]    = parameters["num_factor"]
        if "num_label"     in parameters:
            self.parameters["num_label"]     = parameters["num_label"]
        if "sizes"         in parameters:
            self.parameters["sizes"]         = parameters["sizes"]
        if "hidden_active" in parameters:
            self.parameters["hidden_active"] = parameters["hidden_active"]
        if "output_active" in parameters:   
            self.parameters["output_active"] = parameters["output_active"]
        if "loss"          in parameters:
            self.parameters["loss"]          = parameters["loss"]
        if "learnrate" in parameters:
            self.parameters["learnrate"]     = parameters["learnrate"]
        if "ins_lambda"    in parameters:
            self.parameters["ins_lambda"]    = parameters["ins_lambda"]
        if "label_lambda"  in parameters:
            self.parameters["label_lambda"]  = parameters["label_lambda"]
        

        self.num_feature  = self.parameters["num_feature"]
        self.num_factor   = self.parameters["num_factor"]
        self.num_label    = self.parameters["num_label"]
        self.sizes        = self.parameters["sizes"]
        self.ins_lambda   = self.parameters["ins_lambda"]
        self.label_lambda = self.parameters["label_lambda"] 

        self.num = [ self.num_feature ]
        for i in xrange(len(self.sizes)):
            self.num.append(self.sizes[i])
        self.num.append(self.num_factor)

        ##the w and b for instances
        self.b = []
        self.w = []
        self.grad_b = []
        self.grad_w = []
        for idx in xrange(len(self.num)-1):
            fan_in  = self.num[idx]
            fan_out = self.num[idx + 1]
            r = math.sqrt(6.0 /(fan_in+fan_out) )
            w = [ [random.random() * 2 * r - r for j in xrange(self.num[idx+1])] \
                                               for i in xrange(self.num[idx]) ]
            
            b = [  random.random() * 2 * r - r for j in xrange(self.num[idx+1]) ]
            self.w.append(np.array(w))
            self.b.append(np.array(b))
            self.grad_w.append(np.array(w))
            self.grad_b.append(np.array(b))
      
 
        ## the lw and lb for labels
        r = math.sqrt(6.0 / (self.num_label + self.num_factor) )
        self.lb       = np.array([ random.random() * 2 * r - r \
                                   for j in xrange(self.num_label) ]) 
        self.lw       = np.array([ [ random.random() * 2 * r -r \
                                   for j in xrange(self.num_label)] \
                                   for i in xrange(self.num_factor) ] )
        self.grad_lb  = np.copy(self.lb)
        self.grad_lw  = np.copy(self.lw)


        ##at the end, choose the learnrater
        #self.rater = LearnRate(self)
        #self.rater = AdaGrad(self)
        self.rater = AdaDelta(self)        
        

    def check_dimension(self, x, y = None):
        m,n = x.shape
        if n != self.num_feature:
            logger = logging.getLogger(Logger.project_name);
            logger.error("The self.num_feature (%d) != the actual num of "
                         "features (%d)"%(self.num_feature, n));
            raise Exception("The self.num_feature (%d) != the actual num of"
                            " features (%d)"%(self.num_feature, n));    

        if None == y: return True
        m,n = y.shape
        if n != self.num_label:
            logger = logging.getLogger(logger.project_name);
            logger.error("The self.num_label (%d) != the actual num of "
                         "label (%d)"%(self.num_label, n));
            raise Exception("The self.num_label (%d) != the actual num of "
                            "label %d"%(self.num_label, n));
            
        return True

    def update(self, x, y, idx):
        #print 'update'
        #print 'y'
        #print y.todense()
        #print 'idx'
        #print idx.todense()
        self.check_dimension(x, y )
        self.bp(x, y, idx)
        self.rater.update_before_paramupdate(self)
        self.rater.compute_rate(self)
        self.apply()
        self.rater.update_after_paramupdate(self)


    def ff(self, x):
        self.check_dimension(x)
  
        n_ins,_   = x.shape
        n_layer   = len(self.w)
        tmp       = x

        for i in xrange(n_layer):
            if i == 0 and type(x) == type(sp.csr_matrix([[0]])):
                tmp = tmp * self.w[i]
            else:
                tmp  = np.dot(tmp, self.w[i])
            tmp += np.tile(self.b[i], [n_ins,1] )
            #if i != n_layer - 1:
            tmp  = active( tmp, self.parameters["hidden_active"] ) 
 

        output  = np.dot( tmp, self.lw ) 
        output += np.tile(self.lb,[n_ins, 1])  
        output  = active( output, self.parameters["output_active"] )
        
        return output 


    def bp(self, x, y, idx):
        self.check_dimension(x, y)
        #-------------------------------------------------------
        #ff for train 
        #-------------------------------------------------------
        #compute the instance factor
        hidden_output = []
        n,d           = x.shape
        n_layer       = len(self.w)
        hidden        = []
        tmp           = x
        for i in xrange( n_layer ):
            if 0 == i and type(x) == type(sp.csr_matrix([[0]])):
                tmp = tmp * self.w[i]
            else:
                tmp = np.dot( tmp, self.w[i] )
            tmp += np.tile(self.b[i], [n,1] )
           
            check_large_value_in_hidden=''' 
            m1,n1 = tmp.shape
            flag = False
            for ii in xrange(m1):
                for jj in xrange(n1):
                    if tmp[ii,jj] > 500:
                        print ii,jj,tmp[ii,jj]
                        flag = True
            if True == flag:
                print self.w[i]  
                m1,n1 = self.w[i].shape              
                for ii in xrange(m1):
                    for jj in xrange(n1):
                        if self.w[i][ii,jj] > 1:
                            print ii,jj,self.w[i][ii,jj]
                print self.b[i]
            '''        

            tmp  = active( tmp, self.parameters["hidden_active"] )
            hidden_output.append(tmp)
        ins_factor = tmp

        output  = np.zeros(y.shape)
        m,n     = output.shape
        xy = idx.nonzero()
        for k in xrange(len(xy[0])):
            i = xy[0][k]
            j = xy[1][k]
            output[i,j]  = np.dot( ins_factor[i:i+1, :], self.lw[:, j:j+1])
            output[i,j] += self.lb[j]
        output  = active( output, self.parameters["output_active"] )         

        #---------------------------------------------------
        #compute the grad
        #---------------------------------------------------
        num_rates,_ = idx.shape
        grad_type    = self.parameters["output_active"] \
                       + "_" \
                       + self.parameters["loss"]
        output_grad  = grad( output, y, grad_type )

        
        ## compute grad of label_factor and label_bias
        self.grad_lw = np.zeros(self.grad_lw.shape)
        self.grad_lb = np.zeros(self.grad_lb.shape)
        xy = idx.nonzero()
        for k in xrange(len(xy[0])):
                i = xy[0][k]   
                j = xy[1][k]
                self.grad_lw[:,j:j+1] += output_grad[i,j] \
                                         * np.transpose(ins_factor[i:i+1,:])
                self.grad_lb[j]       += output_grad[i,j]
        for j in xrange(n):
            self.grad_lw[:, j:j+1] /= num_rates #sum_up_to_down[j]
            self.grad_lb[j]        /= num_rates #sum_up_to_down[j]
        

        ## compute grad of instance factor
        ins_factor_grad     = np.zeros(ins_factor.shape)
        xy = idx.nonzero()
        for k in xrange(len(xy[0])):
            i = xy[0][k]
            j = xy[1][k]
            ins_factor_grad[i:i+1,:]   += output_grad[i,j] \
                                          * np.transpose(self.lw[:, j:j+1])

        tmp = ins_factor_grad
        for i in xrange( len(self.w) - 1, -1, -1):
            tmp = tmp * grad(tmp, type =  self.parameters["hidden_active"] )
            if 0 == i and type(x) == type(sp.csr_matrix([[0]])):
                self.grad_w[i] = np.transpose(x) * tmp / num_rates
            elif 0 == i:
                self.grad_w[i] = np.dot( np.transpose(x), tmp ) / num_rates
            else:
                self.grad_w[i] = np.dot( np.transpose(hidden_output[i-1]), tmp )\
                                 / num_rates
     
            self.grad_b[i] = np.sum(tmp, 0) / num_rates
            if 0 == i:  continue
            tmp = np.dot( tmp, np.transpose(self.w[i]) )

    
    def apply(self):
        learn_rate   = self.parameters["learnrate"]
        ins_lambda   = self.parameters["ins_lambda"]
        label_lambda = self.parameters["label_lambda"] 
        n_layer      = len(self.w)
        for i in xrange(n_layer):
            self.w[i] -= self.rater.rate_w[i] * ( self.grad_w[i] \
                                                  + 2 * ins_lambda * self.w[i])
            self.b[i] -= self.rater.rate_b[i] * ( self.grad_b[i] \
                                                  + 2 * ins_lambda * self.b[i])
        
        self.lb -= self.rater.rate_lb * (self.grad_lb + 2 * label_lambda * self.lb)
        self.lw -= self.rater.rate_lw * (self.grad_lw + 2 * label_lambda * self.lw)
        
    def revoke(self):        
        ins_lambda   = self.parameters["ins_lambda"]
        label_lambda = self.parameters["label_lambda"] 
        n_layer      = len(self.w)
        for i in xrange(n_layer):
            self.w[i] += self.rater.rate_w[i] * (self.grad_w[i] \
                                                 + 2 * ins_lambda * self.w[i])
            self.b[i] += self.rater.rate_b[i] * (self.grad_b[i]\
                                                 + 2 * ins_lambda * self.b[i])

        self.lw += self.rater.rate_lw * (self.grad_lw\
                                         + 2 * label_lambda * self.lw)
        self.lb += self.rater.rate_lb * (self.grad_lb\
                                         + 2 * label_lambda * self.lb)
        
 
        
