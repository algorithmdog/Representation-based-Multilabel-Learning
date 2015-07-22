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
                grad_w  = np.asarray(model.grad_w[i].data) + 2 * model.ins_lambda * model.w[i][nonzero]
                self.ada_w[i][nonzero] = self.ada_w[i][nonzero] + grad_w * grad_w            
            else:
                grad_w = model.grad_w[i] + 2 * model.ins_lambda * model.w[i]
                self.ada_w[i] = self.ada_w[i] + grad_w * grad_w
            
            grad_b = model.grad_b[i]
            self.ada_b[i] = self.ada_b[i] + grad_b * grad_b            


        #print type(model.grad_lw)        
        if sp.isspmatrix(model.grad_lw):
            nonzero = model.grad_lw.nonzero()
            #t1,t2 = model.grad_lw.shape
            #print "grad_lw sparity",len(nonzero[0])*1.0/t1 /t2
            self.nonzero['l'] = nonzero
            grad_lw = np.asarray(model.grad_lw.data) + 2 * model.label_lambda * model.lw[nonzero]
            self.ada_lw[nonzero] = self.ada_lw[nonzero] + grad_lw * grad_lw      
        else:
            grad_lw = model.grad_lw + 2 * model.label_lambda * model.lw
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



############## Model ##############
class Model:
    def __init__(self):
        nonparamter = 0 
    def __init__(self,  parameters):
        ##The paramters the user must provide
        self.num_feature   = 1
        self.num_label     = 1
        ##The optional parameters
        self.num_factor    = 300
        self.hidden_active = "tanh"
        self.output_active = "sgmoid"
        self.loss          = "negative_log_likelihood"
        self.learnrate     = 0.1
        self.ins_lambda    = 0.001
        self.label_lambda  = 0.001
        self.sizes         = []
        self.sparse_thr    = 1
	#self.sparse_thr    = 0.000000001
	
        if "num_feature" in parameters:
            self.num_feature = parameters["num_feature"]
        if "num_label" in parameters:
            self.num_label = parameters["num_label"] 
        ##The optional parameters
        if "num_factor" in parameters:
            self.num_factor = parameters["num_factor"] 
        if "hidden_active" in parameters:
            self.hidden_active = parameters["hidden_active"]
        if "output_active" in parameters:   
            self.output_active = parameters["output_active"]
        if "loss" in parameters:
            self.loss = parameters["loss"]
        if "learnrate" in parameters:
            self.learnrate = parameters["learnrate"]
        if "ins_lambda" in parameters:
            self.ins_lambda = parameters["ins_lambda"]
        if "label_lambda" in parameters:
            self.label_lambda  = parameters["label_lambda"]
        if "sizes" in parameters:
            self.sizes = parameters["sizes"]
        if "sparse_thr" in parameters:
            self.sparse_thr = parameters["sparse_thr"]
    
        ##the w and b for instances
        structure = [ self.num_feature ]
        for i in xrange(len(self.sizes)):
            structure.append(self.sizes[i])
        structure.append(self.num_factor)
        #print structure       
 
        self.b = []
        self.w = []
        self.grad_b = []
        self.grad_w = []
        for idx in xrange(len(structure)-1):
            fan_in  = structure[idx]
            fan_out = structure[idx + 1]
            r = math.sqrt(6.0 /(fan_in+fan_out) )
            
            w = 2 * r * np.random.random((structure[idx],structure[idx+1])) - r;
            b = 2 * r * np.random.random(structure[idx+1]) - r;  
            self.w.append(w)
            self.b.append(b)

            self.grad_w.append(np.zeros(w.shape))
            self.grad_b.append(np.zeros(b.shape))   

 
        ## the lw and lb for labels
        r = math.sqrt(6.0 / (self.num_label + self.num_factor) )
        self.lw  = 2 * r * np.random.random((self.num_factor, self.num_label)) -r
        self.lb  = 2 * r * np.random.random(self.num_label) - r

        self.grad_lw  = np.zeros(self.lw.shape)
        self.grad_lb  = np.zeros(self.lb.shape)
        ##at the end, choose the learnrater
        #self.rater = LearnRate(self)
        self.rater = AdaGrad(self)
        #self.rater = AdaDelta(self)        
            
        ## the threshold
        self.thrsel = ThresholdSel()    
        
    def save(self,filename):
        wf = open(filename,"w")
        
        #write the activation function
        wf.write("hidden_active\t%s\n"% self.hidden_active)
        wf.write("output_active\t%s\n"% self.output_active)        

        wf.write("num_layers\t%d\n"%(len(self.w)))
        for k in xrange(len(self.w)):
            m,n = self.w[k].shape
            wf.write("%d\t%d\n"%(m,n))
            for r in xrange(m):
                row     = self.w[k][r,:]
                str_row = "\t".join(map(str,row))
                wf.write("%s\n"%str_row)
            bias     = self.b[k]
            str_bias = "\t".join(map(str,bias))
            wf.write("%s\n"%(str_bias))
               
        m,n = self.lw.shape
        wf.write("%d\t%d\n"%(m,n))
        for r in xrange(m):
            row     = self.lw[r,:]
            str_row = "\t".join(map(str,row))
            wf.write("%s\n"%str_row)
        lb = self.lb
        str_lb = "\t".join(map(str,lb))
        wf.write("%s\n"%(str_lb))

        wf.write("thr\t%lf"%self.thrsel.threshold)

        wf.close()

    def load(self, filename):
        self.w = []
        self.b = []
        self.lw = 0
        self.lb = 0
        
        rf = open(filename)

        ##read active function
        line = rf.readline().strip()
        self.hidden_active = line.split("\t",)[1]
        line = rf.readline().strip()
        self.output_active = line.split("\t",)[1]

        ##read w and b
        line = rf.readline().strip()
        num_layers = int(line.split("\t",)[1])
        m = 0 
        n = 0
        for k in xrange(num_layers):        
            line = rf.readline()
            line = line.strip()
            eles = line.split("\t",)
            m = int(eles[0])
            n = int(eles[1])
            w = np.zeros((m,n))
            for i in xrange(m):
                line = rf.readline().strip()
                w[i,:] = np.array(map(float,line.split("\t",)))
            self.w.append(w)

            line = rf.readline().strip()
            self.b.append(np.array(map(float,line.split("\t",))))    

        ##read lw and lb
        line = rf.readline().strip()
        eles = line.split("\t",)
        m = int(eles[0])
        n = int(eles[1])
        self.lw = np.zeros((m,n))
        for i in xrange(m):
            line = rf.readline().strip()
            eles = line.split("\t",)
            self.lw[i,:] = np.array(map(float,eles))
        line = rf.readline().strip()
        eles = line.split("\t",)
        self.lb = np.array(map(float,eles))

        ##compute 
        self.num_feature = self.w[0].shape[0]        
        self.num_factor  = self.lw.shape[0] 
        self.num_label   = self.lw.shape[1]

        line = rf.readline()
        line.strip()
        self.thrsel           = ThresholdSel()
        self.thrsel.threshold = float(line.split("\t",)[1])

        rf.close()

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

    def printself(self):
        return ;
        print "w[0]"
        m,n = self.w[0].shape
        for i in xrange(m):
            for j in xrange(n):
                print self.w[0][i][j],"\t",
            print ""
        print "_______________"
        print "grad_w[0]"
        m,n = self.w[0].shape
        print self.w[0].shape
        print self.grad_w[0].shape
        tmp = np.asarray(self.grad_w[0].todense())
        for i in xrange(m):
            for j in xrange(n):
                print tmp[i][j],"\t",
            print ""
        print "_______________"


        print "b"
        for i in xrange(n):
                print self.b[0][i],"\t",
        print "_______________"
        print "grad_b"
        for i in xrange(n):
                print self.grad_b[0][i],"\t",
        print "_______________"


        print "lw"
        m,n = self.lw.shape
        for i in xrange(m):
            for j in xrange(n):
                print self.lw[i][j],"\t",
            print ""
        print "_______________"        
        print "grad_lw"
        m,n = self.lw.shape
        tmp = np.asarray(self.grad_lw.todense())
        
        for i in xrange(m):
            for j in xrange(n):
                print tmp[i][j],"\t",
            print ""
        print "_______________"




        print "lb"
        for i in xrange(n):
                print self.lb[i],"\t",
        print "_______________"
        print "grad_lb"
        for i in xrange(n):
                print self.grad_lb[i],"\t",
        print "_______________"



        print "__________________________________________________"
        print "__________________________________________________"


    def update(self, x, y, idx):
        
        #import cProfile, pstats, StringIO
        #pr =  cProfile.Profile()
        #pr.enable()
 
        self.check_dimension(x, y )
        self.bp(x, y, idx)
        self.printself()
        self.rater.update_before_paramupdate(self)
        self.rater.compute_rate(self)
        self.apply()
        self.printself()
        self.rater.update_after_paramupdate(self)
        
        #pr.disable()
        #s = StringIO.StringIO()
        #sortby = 'cumulative'
        #ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
        #ps.print_stats()
        #print "update",s.getvalue()


    def ff(self, x):
        self.check_dimension(x)
  
        n_ins,_   = x.shape
        n_layer   = len(self.w)
        tmp       = x

        for i in xrange(n_layer):
            if i == 0 and sp.isspmatrix(x):
                tmp = tmp * self.w[i]
            else:
                tmp  = np.dot(tmp, self.w[i])
            tmp += np.tile(self.b[i], [n_ins,1] )
            tmp  = active( tmp, self.hidden_active) 
 
        #print tmp.shape
        #print self.lw.shape
        output  = np.dot( tmp, self.lw ) 
        output += np.tile(self.lb,[n_ins, 1])  
        #output  = active( output, self.output_active )
        #output   = np.zeros((n_ins,self.num_label))
        
        return output 


    def bp(self, x, y, idx):

        
        #import cProfile, pstats, StringIO
        #pr =  cProfile.Profile()
        #pr.enable()
        self.check_dimension(x, y)
        #-------------------------------------------------------
        #ff for train 
        #-------------------------------------------------------
        #compute the instance factor
        hidden_output = []
        n,d           = x.shape
        n_layer       = len(self.w)
        #print "n_layer",n_layer
        hidden        = []
        tmp           = x
        for i in xrange( n_layer ):
            if 0 == i and sp.isspmatrix(x):#type(x) == type(sp.csr_matrix([[0]])):
                tmp = tmp * self.w[i]
            else:
                tmp = np.dot( tmp, self.w[i] )
            tmp += np.tile(self.b[i], [n,1] )
            tmp  = active( tmp, self.hidden_active )
            hidden_output.append(tmp)
        ins_factor = tmp

        xy    = idx.nonzero()
        mo,no = idx.shape
        #print "sparity of label", len(xy[0]) * 1.0 / (mo * no)
        if len(xy[0]) * 1.0 / (mo * no) < self.sparse_thr:
            row = ins_factor[xy[0],:]
            col = self.lw[:,xy[1]]
            data  = np.einsum('ik,ki->i',row,col)
            data += self.lb[xy[1]]
	   
            #data = np.zeros(len(xy[0]))            
            #for k in xrange(len(xy[0])):
            #    i = xy[0][k]
            #    j = xy[1][k]
            #    data[k] = np.dot(ins_factor[i,:], self.lw[:,j]) + self.lb[j]
            
            output = sp.csr_matrix((data,xy), idx.shape)

        else: 
            output = np.zeros(idx.shape) 
            for k in xrange(len(xy[0])):
                i = xy[0][k]
                j = xy[1][k]
                output[i,j]  = np.dot( ins_factor[i:i+1, :], self.lw[:, j:j+1]) \
                               + self.lb[j]
        output  = active( output, self.output_active )         

        #---------------------------------------------------
        #compute the grad
        #---------------------------------------------------
        grad_type    = self.output_active \
                       + "_" + self.loss
        output_grad  = grad( output, y, grad_type )
        #print "type(output_grad",type(output_grad),sp.isspmatrix(output_grad) 

        num_rates, _ = idx.shape 
        if sp.isspmatrix(output_grad):
            #print "enter"
            self.grad_lw  = sp.csr_matrix(np.transpose(ins_factor)) * output_grad
            self.grad_lb  = np.asarray(output_grad.sum(0))[0,:]
            #print "sp.issmatrix(self.grad_lw)", sp.isspmatrix(self.grad_lw)
        else:
            self.grad_lw  = self.lw - self.lw
            self.grad_lb  = self.lb - self.lb
            xy = idx.nonzero()
            for k in xrange(len(xy[0])):
                i = xy[0][k]   
                j = xy[1][k]
                self.grad_lw[:,j] += output_grad[i,j] * ins_factor[i,:]
                self.grad_lb[j]   += output_grad[i,j]
        self.grad_lw /= num_rates
        self.grad_lb /= num_rates
	#self.grad_lw = np.asarray(self.grad_lw.todense())
	#self.grad_lb = self.grad_lb.todense()

        ## compute grad of instance factor
        xy = idx.nonzero()
        ins_factor_grad = ins_factor - ins_factor
        #ins_factor_grad = np.zeros(ins_factor.shape)
        if sp.isspmatrix(output_grad):
            for i,j,v in zip(xy[0], xy[1], output_grad.data):
                ins_factor_grad[i,:] += v * self.lw[:,j]
        
        else:
            for k in xrange(len(xy[0])):
                i = xy[0][k]
                j = xy[1][k]
                ins_factor_grad[i,:] += output_grad[i,j] * self.lw[:, j]

       # self.grad_lw = np.asarray(self.grad_lw.todense())
       

        #import cProfile, pstats, StringIO
        #pr =  cProfile.Profile()
        #pr.enable()
        tmp = ins_factor_grad
        for i in xrange( len(self.w) - 1, -1, -1):
            tmp = tmp * grad(tmp, grad_type =  self.hidden_active )
            t1, t2 = x.shape
            if 0 == i and sp.isspmatrix(x):
                #print "sparity of input", len(x.nonzero()[0]) * 1.0 / (t1 * t2)
                if len(x.nonzero()[0]) * 1.0 /(t1 * t2) < self.sparse_thr:
                    self.grad_w[i] = np.transpose(x) * sp.csr_matrix(tmp) / num_rates
                    #self.grad_w[i] = np.asarray(self.grad_w[i].todense())
                    self.grad_w[i] = self.grad_w[i].tocsr()
                else:
                    self.grad_w[i] = np.transpose(x) * tmp / num_rates
                    self.grad_w[i] = self.grad_w[i].tocsr()
            elif 0 == i:
                self.grad_w[i] = np.dot( np.transpose(x), tmp ) / num_rates
            else:
                self.grad_w[i] = np.dot( np.transpose(hidden_output[i-1]), tmp )\
                                 / num_rates
     
            self.grad_b[i] = np.sum(tmp, 0) / num_rates
            if 0 == i:  continue
            tmp = np.dot( tmp, np.transpose(self.w[i]) )
        #pr.disable()
        #s = StringIO.StringIO()
        #sortby = 'cumulative'
        #ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
        #ps.print_stats()
        #print s.getvalue()

    
    def apply(self):
        #import cProfile, pstats, StringIO
        #pr =  cProfile.Profile()
        #pr.enable()
        learn_rate   = self.learnrate
        ins_lambda   = self.ins_lambda
        label_lambda = self.label_lambda
        n_layer      = len(self.w)
        for i in xrange(n_layer):
        #    print type(self.grad_w[i])
        #    print "sp.issmatrix(self.grad_w[i]",sp.isspmatrix(self.grad_w[i])
            if sp.isspmatrix(self.grad_w[i]):
                nonzero             = self.rater.nonzero['%d'%i]
        #        print "nonzero_grad",self.grad_w[i].nonzero()
                self.w[i][nonzero] -= self.rater.rate_w[i][nonzero] * ( np.asarray(self.grad_w[i].data) \
                                      + 2 * ins_lambda * self.w[i][nonzero])
                #self.w[i]  -= self.rater.rate_w[0] * (np.asarray(self.grad_w[i].todense()) + 2 * ins_lambda * self.w[i])
               # print "w[0]"
               # m,n = self.w[0].shape
                #for k in xrange(m):
                #    for j in xrange(n):
                #       print self.w[0][k][j],"\t",
                 #   print ""

            else:
                self.w[i] -= self.rater.rate_w[i] * (self.grad_w[i] + 2 * ins_lambda * self.w[i])
            self.b[i] -= self.rater.rate_b[i] * ( self.grad_b[i] + 2 * ins_lambda * self.b[i])
       
       
        if sp.isspmatrix(self.grad_lw):
            nonzero           = self.rater.nonzero['l']
            self.lw[nonzero] -= self.rater.rate_lw[nonzero] * (np.asarray(self.grad_lw.data) \
                                + 2 * label_lambda * self.lw[nonzero])
            #self.lw -= self.rater.rate_lw * (np.asarray(self.grad_lw.todense()) + 2 * label_lambda * self.lw)
        else:
            self.lw -= self.rater.rate_lw * (self.grad_lw + 2 * label_lambda * self.lw) 
        self.lb -= self.rater.rate_lb * (self.grad_lb + 2 * label_lambda * self.lb)
    
        #pr.disable()
        #s = StringIO.StringIO()
        #sortby = 'cumulative'
        #ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
        #ps.print_stats()
        #print "apply",s.getvalue()
