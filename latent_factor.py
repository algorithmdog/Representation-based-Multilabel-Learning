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

        num_feature = self.parameters["num_feature"];
        num_factor  = self.parameters["num_factor"];
        num_label   = self.parameters["num_label"];
        sizes       = self.parameters["sizes"];

        num = [num_feature];
        for i in sizes:
            num.append(i);
        num.append(num_factor);

        self.b = [];
        self.w = [];
        self.grad_b = [];
        self.grad_w = [];
        for idx in len(num)-1:
            w = [ [0.0 for j in num[idx+1]] for i in num[idx] ];
            b = [  0.0 for j in num[idx+1 ];
            self.w.append(np.array(w));
            self.b.append(np.array(b));
            self.grad_w.append(np.array(w));
            self.grad_b.append(np.array(b));
       
        self.lb       = np.array([ 0.0 for j in num_label ]); 
        self.grad_lb  = np.copy(self.lb);
        label_factors = [ [0.0 for j in xrange(num_label)] for i in xrange(num_factor) ];
        self.lf       = np.array(label_factors);
        self.grad_lf  = np.copy(self.lf);

    def check_dimenson(self, x, y = None):
        m,n = x.shape;
        if n != self.num_feature:
            Logger.instance.error("The self.num_feature %d != the actual num of \
                                   features %d"%(self.num_feature, n));
            raise Exception("The self.num_feature %d != the actual num of features %d"\
                             %(self.num_feature, n));    

        if None == y:   return;
        m,n = y.shape;
        if n != self.num_label:
            Logger.instance.error("The self.num_label %d != the actual num of label %d"\
                                  %(self.num_label, n));
            raise Exception("The self.num_label %d != the actual num of label %d"\
                            %(self.num_label, n));
            

    def update(self, x, y, idx):
        check_dimension(x, y );
        self.bp(x, y, idx);
        self.apply();

    def ff(self, x):
        check_dimension(x, y);
        n_layer = len(self.w);
        tmp     = x;
        for i in xrange(len(self.w) - 1):
            tmp  = np.dot(tmp, self.w[i]);
            tmp += np.unkownfunc(self.b[i]);
            tmp  = active( tmp, self.parameters["hidden_active"] ); 
        tmp  = np.dot(tmp. self.w[n_layer-1]);
        tmp += np.unkownfunc(self.b[n_layer-1]);

        output  = np.dot( tmp, self.lf ); 
        output += np.unkownfunc(self.lb);      
        output  = active( output, self.parameters["output_active"] );
        return output; 


    def bp(self, x, y, idx):
        check_dimension(x, y);
        ##ff for train 
        n,d     = x.shape;
        n_layer = len(self.w);
        hidden    = [];
        tmp       = x;
        for i in xrange( n_layer-1 );
            tmp  = np.dot( tmp, self.w[i] );
            tmp += np.unkownfunc(self.b);
            tmp  = active( tmp, self.parameters["hidden_active"] );
            hidden_output.append(tmp);

        ins_factor  = np.dot( tmp, self.w[ n_layer - 1 ] );
        ins_factor += np.unkownfunc( self.b[ n_layer - 1] );
        output  = np.zeros(y.shape);
        m,n     = output.shape;
        for i in xrange(m):
            for j in xrange(n):
                if 0 == idx[i,j]: continue;
                output[i,j]  = np.dot( ins_factor[i:i+1, :], self.lf[:, j:j+1]);
                output[i,j] += self.lb[j];
        output  = active( output, self.parameters["output_active"] );         

        ##compute grad
        grad_type    = self.parameters["output_active"] \
                       + "_" \
                       + self.parameters["loss"];
        output_grad  = grad( output, y, grad_type, idx);

        sum_up_to_down      = np.sum(idx,0);
        sum_left_to_right   = np.sum(idx,1);
        self.grad_lf = np.zeros(self.grad_lf.shape);
        self.grad_lb = np.zeros(self.grad_lb.shape);
        ins_f_grad   = np.zeros(ins_factor.shape);
        for i in xrange(m):
            for j in xrange(n):
                if 0 == idx[i,j]: continue;
                self.grad_lf[:,j:j+1] += output_grad[i,j] \
                                         * np.transpose(ins_factor[i:i+1,:]);
                self.grad_lb[j]       += output_grad[i,j]; 
                ins_f_grad[i:i+1,:]   += output_grad[i,j] \
                                         * np.transpose(self.lf[:, j:j+1]);
        for i in xrange(m):
            if 0 == sum_left_to_right[i]:   continue;
            ins_f_grad[i:i+1,:] /= sum_left_to_right[i];
        for j in xrange(n):
            if 0 == sum_up_to_down[j]:  continue;
            self.grad_lf[:, j:j+1] /= sum_up_to_down[j];
            self.grad_lb[j]        /= sum_up_to_down[j];

 
        tmp = ins_factor_grad;
        for i in xrange( len(self.w) - 1, -1, -1):
            if 0 == i:
                self.grad_w[i] = np.dot( np.transpose(x), tmp ) / n;
            else:
                self.grad_w[i] = np.dot( np.transpose(hidden_output[i-1]), tmp )\
                                 / n;
            self.grad_b[i] = np.sum(tmp, 0) / n;
            tmp = np.dot( tmp, np.transpose(self.w[i]) );
 
    def apply(self):
        ins_lambda   = self.parameters["ins_lambda"];
        label_lambda = self.parameters["label_lambda"]; 
        n_layer      = len(self.w);
        for i in xrange(n_layer):
            self.w[i] -= ins_lambda * self.grad_w[i];
            self.b[i] -= ins_lambda * self.grad_b[i];
        
        self.lb -= label_lambda * self.grad_lb;
        self.lf -= label_lambda * self.grad_lf;
        
    def revoke(self):        
        ins_lambda   = self.parameters["ins_lambda"];
        label_lambda = self.parameters["label_lambda"]; 
        n_layer      = len(self.w);
        for i in xrange(n_layer):
            self.w[i] += ins_lambda * self.grad_w[i];
            self.b[i] += ins_lambda * self.grad_b[i];
        self.lf += label_lambda * self.grad_lf;
        self.lb += label_lambda * self.grad_lb;
        
 
        
