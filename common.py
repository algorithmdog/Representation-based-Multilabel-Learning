#!/bin/bash

import logging, Logger

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


act      = Enum(["sgmoid","tanh","linear"])
lo       = Enum(["negative_log_likelihood","least_square"])
grad = Enum(["sgmoid_negative_log_likelihood",\
             "linear_least_square",\
             "sgmoid",\
             "tanh",\
             "linear"]);

ha_map = {0:act.tanh,   1:act.linear};
oa_map = {0:act.sgmoid, 1:act.linear};
lo_map = {0:lo.negative_log_likelihood, 1:lo.least_square}

def actlo2grad(actid, loid):
    if actid == act.sgmoid and loid == lo.negative_log_likelihood:
        return grad.sgmoid_negative_log_likelihood
    elif actid == act.linear and loid == lo.least_square:
        return grad.linear_least_square
    else:
        actname  = act_name[actid]
        lossname = lo_name[loid]
        logger   = logging.getLogger(Logger.project_name)
        logger.error("Activation (%s) and loss (%s) lead to no grad"%(actname,lossname))
        raise Exception("Activation (%s) and loss (%s) lead to no grad"%(actname,lossname))


op     = Enum(["gradient","alternative_least_square"]);
op_map = {0:op.gradient, 1:op.alternative_least_square}

st     = Enum(["full_sampler","instance_sampler"]);
st_map = {0:st.full_sampler, 1:st.instance_sampler} 

m      = Enum(["internal_memory", "external_memory"])
m_map  = {0: m.internal_memory, 1:m.external_memory} 


####params 
default_params = dict()
default_params["h"]            = 100
default_params["ha"]           = act.tanh
default_params["oa"]           = act.sgmoid

default_params["l"]            = lo.negative_log_likelihood
default_params["l2"]           = 0.001
default_params["b"]            = 10
default_params["i"]            = 20
default_params["st"]           = st.instance_sampler
default_params["sr"]           = 5 
default_params["sp"]           = 0.01
default_params["op"]           = op.gradient
default_params["m"]            = m.internal_memory    
default_params["r"]            = 0.1

default_params["sizes"]        = []


