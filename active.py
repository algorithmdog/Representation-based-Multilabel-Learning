#!/bin/python
import os;
import sys;

path      = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/../utils/Python_Utils");
sys.path.append(path + "/../Python_Utils");

import numpy as np;
import Logger;

#active function list
#1. linear ( not active )
#2. sgmoid
#3. tanh
def active(A, active_type="sgmoid", idx = None):
    if "linear" == active_type:
        None;
    elif "sgmoid" == active_type:
        A = 1 / ( 1 + 1/np.exp(A) );
    elif "tanh" == active_type:
        ex  = np.exp(A);
        enx = np.exp(-A);
        A   = (ex - enx) / (ex + enx);
    else:
        Logger.instance.error("Not recognized active function %s"%active_type);
        raise Exception("Not recognized active function %s"%active_type);
    return A;


#@loss_type: the loss function list
#1. negative log likelihood
#2. least square 
#3. appro l1 hinge
#4. l2 hinge
#@parameters. The parameter of loss function. For example, the regularization coefficient
def loss(A, Y, loss_type = "negative_log_likelihood", idx = None):
    if   "negative_log_likelihood" == loss_type:
        return np.sum( -Y * np.log(A) - (1 - Y) * np.log(1-A) );
    elif "least_square"            == loss_type:
        return np.sum( (Y - A) * (Y - A) );
    elif "appro_l1_hinge"          == loss_type:
        loss = 1 - Y * A;
        m,n = loss.shape;
        for i in xrange(m):
            for j in xrange(n):
                if   loss[i,j]  < 0.0: loss[i,j] = 0.0;
                elif loss[i,j]  > 0.0 and loss[i,j] < 1.0: 
                    square    = loss[i,j];
                    loss[i,j] = loss[i,j] * square - square - loss[i,j] + 1;
        return np.sum(loss);
    elif "l2_hinge"                == loss_type:
        return np.sum( np.zeros(A.shape), 1 - Y*A );         
    else:
        Logger.instance.error("Not recognized loss function %s"%loss_type);
        raise Exception("Not recognized loss function %s"%loss_type);
    


#@type. the grad() computes the  gradient of the type function. \
#       The type function may be the loss function or the active function 
# 1. sgmoid active and negative log likelihood loss
# 2. least square loss
# 3. approximate l1 hinge loss
# 4. l2 hinge loss 
# 5. sgmoid
# 6. linear
# 7. tanh
#@parameters. 
def grad(A, Y = None, type = " sgmoid_negativeloglikelihood ", idx = None):
    if ("sgmoid_negativeloglikelihood" == type or \
        "linear_least_square"          == type or \
        "linear_approx_l1_hinge"       == type or \
        "linear_l2_hinge"              == type    \
       ) and None == Y:   
        Logger.instance.error("None == Y when computing gradients of loss function %s"%type);
        raise Exception("None == Y when computing gradients of loss function %s"%type);
       
    if   "sgmoid_negativeloglikelihood" == type:
        return A - Y;        
    elif "linear_least_square"                 == type:
        return 2*(A - Y);
    elif "linear_appro_l1_hinge"               == type:
        Y = 2 * Y - 1;
        score = Y * A;
        grad = np.zeros(score.shape);
        m,n  = score.shape;
        for i in xrange(m):
            for j in xrange(n):
                if   score[i,j] > 1.0:
                    grad[i,j] = 0.0;
                elif score[i,j] > 0.0 and score[i,j] < 1.0:
                    grad[i,j] = 6 * Y[i,j] * Y[i,j] * A[i,j]  - 2 * Y[i,j];
                else:
                    grad[i,j] = -Y[i,j];
        return grad;
    elif "linear_l2_hinge"                     == type:
        Y = 2 * Y - 1;
        score = Y * A;
        grad  = np.zeros(Y.shape);
        m,n   = score.shape;
        for i in xrange(m):
            for j in xrange(n):
                if score[i,j] < 1:
                    grad[i,j] = -Y[i,j];
        return grad;
    elif "sgmoid"                       == type:
        return A * (1 - A)
    elif "linear"                       == type:
        return np.ones(A.shape);
    elif "tanh"                         == type:
        return 1 - A * A;  
    else:
        Logger.instance.info("Not recognized grad target function %s"%type);
        raise Exception("Not recognized grad target function %s"%type);

