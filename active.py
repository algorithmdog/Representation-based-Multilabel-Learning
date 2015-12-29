#!/bin/python
import os;
import sys;

path      = os.path.split(os.path.realpath(__file__))[0];

from common import *
import math
import numpy as np
import scipy.sparse as sp
import logging,Logger

#active function list
#1. linear ( not active )
#2. sgmoid
#3. tanh
#4. rel
def active(A, active_type= act.sgmoid, idx = None):
    if act.linear == active_type:
        None;
    elif act.sgmoid == active_type:
        if sp.isspmatrix(A):#type(A) == type(sp.csr_matrix([[0]])):
            A.data = 1 / ( 1 + 1 / np.exp(A.data))
        else:
            A = 1 / ( 1 + 1/np.exp(A) );
    elif act.tanh == active_type:
        A = np.tanh(A)
    elif "rel" == active_type:
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
#1. negative_log_likelihood
#2. least_square 
#3. appro_l1_hinge
#4. l2_hinge
def loss(A, Y, loss_type = lo.negative_log_likelihood, idx = None):
    #if idx not specified, full one
    if None == idx:
        idx = np.ones(Y.shape);
        

    #negative_log_likelihood
    #loss = -ylog(f(x)) - (1-y)log(1-f(x))
    if lo.negative_log_likelihood == loss_type:
        return np.sum( -Y * np.log(A) * idx - (1 - Y) * np.log(1-A) * idx );


    # least_square
    # loss = (y-f(x))^2
    elif lo.least_square == loss_type:
        return np.sum( (Y - A) * (Y - A) * idx );


    # appro_l1_hinge
    # y should be in {-1,1}, so 2*y-1;
    #       |-   0;          1-yf(x) < 0 
    #       |
    #loss = |    (yf(x))^3 - (yf(x))^2 - (yf(x)) +1;  0 <= 1-yf(x) < 1
    #       |
    #       |-   1-yf(x);    1-yf(x) >= 1
    elif "appro_l1_hinge" == loss_type:
        scores = 1 - (2*Y-1) * A;
        loss   = 0;
        m,n    = scores.shape;
        for i in xrange(m):
            for j in xrange(n):
                if 0 == idx[i,j]: continue;
                if scores[i,j] < 0.0:   loss += 0;
                elif scores[i,j] >= 0.0 and scores[i,j] <1.0:
                        yfx     = 1 - scores[i,j];
                        square  = yfx * yfx;
                        loss   += yfx * square - square - yfx + 1;
                else:
                        loss += scores[i,j];
        return loss;

    #l2_hinger
    # y should be in {-1,1}, so 2*y-1
    #       |-   0                1 - yf(x) <= 0
    #loss = |
    #       |-   (1-yf(x))^2      1 - yf(x)  > 0
    elif "l2_hinge" == loss_type:
        scores = 1 - (2*Y-1) * A;
        loss   = 0;
        m,n  = scores.shape;
        for i in xrange(m):
            for j in xrange(n):
                if 0 == idx[i,j]:   continue;
                if scores[i,j] < 0: continue;
                if scores[i,j] > 0: loss += scores[i,j] * scores[i,j];                
        return  loss;
    
    
    else:
        logger = logging.getLogger(Logger.project_name);
        logger.error("Not recognized loss function: %s"%loss_type);
        raise Exception("Not recognized loss function: %s"%loss_type);
    


#@type. the grad() computes the  gradient of the type function. \
#       The type function may be the loss function or the active function 
# 1. sgmoid_negative_log_likelihood
# 2. least_square
# 3. appro_l1_hinge
# 4. l2_hinge
# 5. sgmoid
# 6. linear
# 7. tanh
# 8. rel
#@parameters. 
def grad(A, Y = None, grad_type = " sgmoid_negative_log_likelihood "):
    if ("sgmoid_negative_log_likelihood" == grad_type or \
        "linear_least_square"            == grad_type or \
        "linear_appro_l1_hinge"          == grad_type or \
        "linear_l2_hinge"                == grad_type    \
       ) and None == Y:   
            logger = logging.getLogger(Logger.project_name);
            logger.error("Y should not equals None when computing gradients"
                         " of loss function %s"%grad_type);
            raise Exception("Y should not equal None when computing gradients of loss"
                            " function %s"%grad_type);
       

    if "sgmoid_negative_log_likelihood" == grad_type:
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
    elif "linear_least_square" == grad_type:
        return 2*(A - Y);
    elif "linear_appro_l1_hinge" == grad_type:
        Y = 2 * Y - 1;
        score = Y * A;
        grad = np.zeros(score.shape);
        
        # score < 0 case
        flag = score < 0;
        grad[ flag ]  = -Y[ flag ];
        
        # 0 <= score and score < 1 case
        tmp   = (3 * score * score * Y - 2 * score * Y - Y);
        flag1 = 0 <= score;
        flag2 = score < 1.0;
        flag = flag1 * flag2;
        grad[ flag ] = tmp[ flag ];    
    
        return grad; 
    elif "linear_l2_hinge" == grad_type:
        Y = 2 * Y - 1;
        score = Y * A;
        grad  = np.zeros(Y.shape);
        flag = score <= 1;
        grad[ flag ] =  (-2 * Y * ( 1 - score))[flag];
        return grad;

    elif "sgmoid" == grad_type:
        return A * (1 - A)

    elif "linear" == grad_type:
        return np.ones(A.shape);

    elif "tanh" == grad_type:
        return 1 - A * A;  

    elif "rel" == grad_type:
        A[A <  0] = 0;
        A[A >  0] = 1;
        return A;

    else:
        logger = logging.getLogger(Logger.project_name);
        logger.info("Not recognized grad target function: %s"%grad_type);
        raise Exception("Not recognized grad target function: %s"%grad_type);

