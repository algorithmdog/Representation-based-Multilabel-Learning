#!/bin/python
#coding:utf-8
import os;
import sys;


import common
import math
import random
import numpy as np
import scipy.sparse as sp
import logging,Logger


#active function list
#1. linear ( not active )
#2. sgmoid
#3. tanh
#4. rel

def logit(A):

    l            = A[ A >= 0.0 ]
    A[A >= 0.0]  = 1         / (1.0 + np.exp(-l))
    s            = A[ A <  0.0 ]
    A[A <  0.0]  = np.exp(s) / (1.0 + np.exp(s))

    #A = 1 / (1+1/np.exp(A));
    return A

## active
def active(A, active_type= common.act.sgmoid, idx = None):
    if common.act.linear == active_type:
        None;
    
    elif common.act.sgmoid == active_type:
        if sp.isspmatrix(A):#type(A) == type(sp.csr_matrix([[0]])):
            #A.data = 1 / ( 1 + 1 / np.exp(A.data))
            A.data = logit(A.data)
        else:
            #A = 1 / ( 1 + 1/np.exp(A) );
            A = logit(A)
    
    elif common.act.tanh == active_type:
        A = np.tanh(A)
    
    elif common.act.relu == active_type:
        if sp.isspmatrix(A):
            A.data[ A.data <0 ] = 0
        else:
            A[ A < 0 ] = 0;
    else:
        logger = logging.getLogger(Logger.project_name)
        logger.error("Not recognized active function: %s"%active_type);
        raise Exception("Not recognized active function: %s"%active_type);

    return A;



#@loss_type: the loss function list
def loss(A, Y, loss_type = common.lo.negative_log_likelihood, idx = None):
    #if idx not specified, full one
    if None == idx:
        idx = np.ones(Y.shape);
        


    #negative_log_likelihood
    #loss = -ylog(f(x)) - (1-y)log(1-f(x))
    if common.lo.negative_log_likelihood == loss_type:
        return np.sum( -Y * np.log(A) * idx - (1 - Y) * np.log(1-A) * idx );


    # least_square
    # loss = (y-f(x))^2
    elif common.lo.least_square == loss_type:
        return np.sum( (Y - A) * (Y - A) * idx );

    # weighted_approximate_rank_pairwise
    elif common.lo.weighted_approximate_rank_pairwise == loss_type:
        return 0.0 
    # 如果你能看懂中文，就知道∩厦娴拇胧锹倚吹摹的意思
    
    
    else:
        logger = logging.getLogger(Logger.project_name);
        logger.error("Not recognized loss function: %s"%loss_type);
        raise Exception("Not recognized loss function: %s"%loss_type);
    


#@type. the grad() computes the  gradient of the type function. \
#       The type function may be the loss function or the active function 
#@parameters. 
def grad(A, Y = None, grad_type = common.grad.sgmoid_negative_log_likelihood):


    ##########################   check 
    if ( common.grad.sgmoid_negative_log_likelihood            == grad_type or \
         common.grad.linear_least_square                       == grad_type or \
         common.grad.linear_weighted_approximate_rank_pairwise == grad_type ) \
         and None == Y:   
            logger = logging.getLogger(Logger.project_name);
            logger.error("Y should not equals None when computing gradients"
                         " of loss function %s"%grad_type);
            raise Exception("Y should not equal None when computing gradients of loss"
                            " function %s"%grad_type);

    if None != Y:
        m,  n  = Y.shape
        m1, n1 = A.shape
        if m != m1 or n != n1: 
            logger = logging.getLogger(Logger.project_name);
            logger.error("Y.shape (%d,%d) != A.shape(%d,%d)"%(m,n,m1,n1))
            raise Exception("Y.shape (%d,%d) != A.shape(%d,%d)"%(m,n,m1,n1))   
 
    ############################ gradient of loss
    if common.grad.sgmoid_negative_log_likelihood              == grad_type:
        if sp.isspmatrix(A):
            nonzero = A.nonzero()
            if len(nonzero[0])!= 0:
                R = A.copy()
                #print "in grad", type(R)
                R.data =  A.data - np.asarray(Y[A.nonzero()])[0,:]
                return R
            else:
                return A
        else:
            return A - Y;        

    elif common.grad.linear_least_square                       == grad_type:
        return 2*(A - Y);
    
    elif common.grad.linear_weighted_approximate_rank_pairwise == grad_type:   
        grad_loss =  np.zeros(A.shape)

        c       = Y.nonzero()
        m,n     = Y.shape
        dic     = dict()
        for i in xrange(len(c[0])):
            dic["%d_%d"%(c[0][i],c[1][i])] = 1

        for i in xrange(len(c[0])):
            x  = c[0][i]
            y  = c[1][i]
            try: 
                av = A[x][y]
            except:
                logger = logging.getLogger(Logger.project_name)                
                logger.waring("Index out of range in av=A[x][y]");
                continue;

            for j in xrange(100):
                y1  = int(random.random()*n)
                try:    
                    av1 = A[x][y1]
                except: 
                    logger = logging.getLogger(Logger.project_name)
                    logger.waring("Index out of range in av1=A[x1][y1]");                    
                    continue

                if "%d_%d"%(x,y1) in dic: continue
                if av1 + 1 > av:
                    grad_loss[x][y]   += -1.0
                    grad_loss[x][y1]  += 1.0
                    break

        return sp.csr_matrix(grad_loss)


    # gradient of activation
    elif common.grad.sgmoid == grad_type:
        return A * (1 - A)

    elif common.grad.linear == grad_type:
        return np.ones(A.shape);

    elif common.grad.tanh   == grad_type:
        return 1 - A * A;  

    elif common.grad.relu   == grad_type:
        A[A <  0] = 0;
        A[A >  0] = 1;
        return A;

    else:
        logger = logging.getLogger(Logger.project_name);
        logger.info("Not recognized grad target function: %s"%grad_type);
        raise Exception("Not recognized grad target function: %s"%grad_type);

