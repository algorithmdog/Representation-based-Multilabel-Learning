#!/bin/python
import os;
import sys;

path = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path);
sys.path.append(path + "/../utils/Python_Utils");
sys.path.append(path + "/../Python_Utils");

from active import *;
import numpy as np;
import math;
import pickle;
import random;

import logging, Logger;

class Model:
    def __init__(self,  parameters):
        self.parameters = dict();
        self.parameters["num_feature"]   = 100;
        self.parameters["num_factor"]    = 100;
        self.parameters["num_label"]     = 1000;
        self.parameters["sizes"]         = [100];
        self.parameters["hidden_active"] = "tanh";
        self.parameters["output_active"] = "sgmoid";
        self.parameters["loss"]          = "negative_log_likelihood";
        self.parameters["ins_lambda"]    = 0.001;
        self.parameters["label_lambda"]  = 0.001;
        
        if "num_feature" in parameters:
            self.parameters["num_feature"]   = parameters["num_feature"];
        if "num_factor"    in parameters:
            self.parameters["num_factor"]    = parameters["num_factor"];
        if "num_label"     in parameters:
            self.parameters["num_label"]     = parameters["num_label"];
        if "sizes"         in parameters:
            self.parameters["sizes"]         = parameters["sizes"];
        if "hidden_active" in parameters:
            self.parameters["hidden_active"] = parameters["hidden_active"];
        if "output_active" in parameters:   
            self.parameters["output_active"] = parameters["output_active"];
        if "loss"          in parameters:
            self.parameters["loss"]          = parameters["loss"];
        if "ins_lambda"    in parameters:
            self.parameters["ins_lambda"]    = parameters["ins_lambda"];
        if "label_lambda"  in parameters:
            self.parameters["label_lambda"]  = parameters["label_lambda"];

        self.num_feature = self.parameters["num_feature"];
        self.num_factor  = self.parameters["num_factor"];
        self.num_label   = self.parameters["num_label"];
        self.sizes       = self.parameters["sizes"];

        self.num = [ self.num_feature ];
        for i in xrange(len(self.sizes)):
            self.num.append(self.sizes[i]);
        self.num.append(self.num_factor);

        ##the w and b for instances
        self.b = [];
        self.w = [];
        self.grad_b = [];
        self.grad_w = [];
        for idx in xrange(len(self.num)-1):
            fan_in  = self.num[idx]
            fan_out = self.num[idx + 1]
            r = math.sqrt(6.0 /(fan_in+fan_out) );
            w = [ [random.random() * 2 * r - r for j in xrange(self.num[idx+1])] \
                                               for i in xrange(self.num[idx]) ];
            
            b = [  random.random() * 2 * r - r for j in xrange(self.num[idx+1]) ];
            self.w.append(np.array(w));
            self.b.append(np.array(b));
            self.grad_w.append(np.array(w));
            self.grad_b.append(np.array(b));
      
 
        ## the lw and lb for labels
        r = math.sqrt(6.0 / (self.num_label + self.num_factor) );
        self.lb       = np.array([ random.random() * 2 * r - r \
                                   for j in xrange(self.num_label) ]); 
        self.lw       = np.array([ [ random.random() * 2 * r -r \
                                   for j in xrange(self.num_label)] \
                                   for i in xrange(self.num_factor) ] );
        self.grad_lb  = np.copy(self.lb);
        self.grad_lw  = np.copy(self.lw);

    def check_dimension(self, x, y = None):
        m,n = x.shape;
        if n != self.num_feature:
            logger = logging.getLogger(Logger.project_name);
            logger.error("The self.num_feature (%d) != the actual num of "
                         "features (%d)"%(self.num_feature, n));
            raise Exception("The self.num_feature (%d) != the actual num of"
                            " features (%d)"%(self.num_feature, n));    

        if None == y: return True;
        m,n = y.shape;
        if n != self.num_label:
            logger = logging.getLogger(logger.project_name);
            logger.error("The self.num_label (%d) != the actual num of "
                         "label (%d)"%(self.num_label, n));
            raise Exception("The self.num_label (%d) != the actual num of "
                            "label %d"%(self.num_label, n));
            
        return True;

    def update(self, x, y, idx):
        self.check_dimension(x, y );
        self.bp(x, y, idx);
        self.apply();

    def ff(self, x):
        self.check_dimension(x);
  
        n_ins,_   = x.shape;
        n_layer   = len(self.w);
        tmp       = x;

        for i in xrange(len(self.w) - 1):
            tmp  = np.dot(tmp, self.w[i]);
            tmp += np.tile(self.b[i], [n_ins,1] );
            tmp  = active( tmp, self.parameters["hidden_active"] ); 
        tmp  = np.dot(tmp, self.w[n_layer-1]);
        tmp += np.tile(self.b[n_layer-1], [n_ins,1] );

        output  = np.dot( tmp, self.lw ); 
        output += np.tile(self.lb,[n_ins, 1]);  
        output  = active( output, self.parameters["output_active"] );
        
        return output; 


    def bp(self, x, y, idx):
        self.check_dimension(x, y);
        #-------------------------------------------------------
        #ff for train 
        #-------------------------------------------------------
        #compute the instance factor
        hidden_output = [];
        n,d           = x.shape;
        n_layer       = len(self.w);
        hidden        = [];
        tmp           = x;
        for i in xrange( n_layer-1 ):
            tmp  = np.dot( tmp, self.w[i] );
            tmp += np.tile(self.b[i], [n,1] );
            tmp  = active( tmp, self.parameters["hidden_active"] );
            hidden_output.append(tmp);
        ins_factor  = np.dot( tmp, self.w[ n_layer - 1 ] );
        ins_factor += np.tile( self.b[ n_layer - 1], [n,1] );

        ## compute the output
        output  = np.zeros(y.shape);
        m,n     = output.shape;
        for i in xrange(m):
            for j in xrange(n):
                if 0 == idx[i,j]: continue;
                output[i,j]  = np.dot( ins_factor[i:i+1, :], self.lw[:, j:j+1]);
                output[i,j] += self.lb[j];
        output  = active( output, self.parameters["output_active"] );         

        ##compute grad of instance factor and label_factor(lw);
        grad_type    = self.parameters["output_active"] \
                       + "_" \
                       + self.parameters["loss"];
        output_grad  = grad( output, y, grad_type );

        ## compute grad of instance factor
        sum_up_to_down      = np.sum(idx,0);
        sum_left_to_right   = np.sum(idx,1);
        ins_factor_grad   = np.zeros(ins_factor.shape);
        for i in xrange(m):
            for j in xrange(n):
                if 0 == idx[i,j]: continue;
                ins_factor_grad[i:i+1,:]   += output_grad[i,j] \
                                              * np.transpose(self.lw[:, j:j+1]);
        
        #for i in xrange(m):
        #    if 0 == sum_left_to_right[i]:   continue;
        #    ins_factor_grad[i:i+1,:] /= sum_left_to_right[i];

        ## compute grad of label_factor and label_bias
        self.grad_lw = np.zeros(self.grad_lw.shape);
        self.grad_lb = np.zeros(self.grad_lb.shape);
        for i in xrange(m):
            for j in xrange(n):
                if 0 == idx[i,j]: continue;
                self.grad_lw[:,j:j+1] += output_grad[i,j] \
                                         * np.transpose(ins_factor[i:i+1,:]);
                self.grad_lb[j]       += output_grad[i,j];
        for j in xrange(n):
            if 0 == sum_up_to_down[j]:  continue;
            self.grad_lw[:, j:j+1] /= sum_up_to_down[j];
            self.grad_lb[j]        /= sum_up_to_down[j];
        

        ## compute grad of w and b
        num_rates = 0;
        for i in xrange(len(sum_up_to_down)):
            num_rates += sum_up_to_down[i];

        tmp = ins_factor_grad;
        for i in xrange( len(self.w) - 1, -1, -1):
            if 0 == i:
                self.grad_w[i] = np.dot( np.transpose(x), tmp ) / num_rates;
            else:
                self.grad_w[i] = np.dot( np.transpose(hidden_output[i-1]), tmp )\
                                 / num_rates;
     
            self.grad_b[i] = np.sum(tmp, 0) / num_rates;
            if 0 == i:  continue;
            tmp = np.dot( tmp, np.transpose(self.w[i]) );
    
    def apply(self):
        ins_lambda   = self.parameters["ins_lambda"];
        label_lambda = self.parameters["label_lambda"]; 
        n_layer      = len(self.w);
        for i in xrange(n_layer):
            self.w[i] -= ins_lambda * self.grad_w[i];
            self.b[i] -= ins_lambda * self.grad_b[i];
        
        self.lb -= label_lambda * self.grad_lb;
        self.lw -= label_lambda * self.grad_lw;
        
    def revoke(self):        
        ins_lambda   = self.parameters["ins_lambda"];
        label_lambda = self.parameters["label_lambda"]; 
        n_layer      = len(self.w);
        for i in xrange(n_layer):
            self.w[i] += ins_lambda * self.grad_w[i];
            self.b[i] += ins_lambda * self.grad_b[i];
        self.lw += label_lambda * self.grad_lw;
        self.lb += label_lambda * self.grad_lb;
        
 
        
