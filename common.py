#!/bin/bash

import logging, Logger
import copy

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


act      = Enum(["sgmoid","tanh","linear","relu"])
lo       = Enum(["negative_log_likelihood","least_square","weighted_approximate_rank_pairwise"])
grad     = Enum(["sgmoid_negative_log_likelihood",\
                 "linear_least_square",\
                 "linear_weighted_approximate_rank_pairwise",\
                 "sgmoid",\
                 "tanh",\
                 "linear",\
                 "relu"]);

ha_map = {0:act.tanh,   1:act.linear, 2:act.relu};
oa_map = {0:act.sgmoid, 1:act.linear};
lo_map = {0:lo.negative_log_likelihood, 1:lo.least_square, 2:lo.weighted_approximate_rank_pairwise}

def actlo2grad(actid, loid):
    if actid == act.sgmoid and loid == lo.negative_log_likelihood:
        return grad.sgmoid_negative_log_likelihood
    elif actid == act.linear and loid == lo.least_square:
        return grad.linear_least_square
    elif actid == act.linear and loid == lo.weighted_approximate_rank_pairwise:
        return grad.linear_weighted_approximate_rank_pairwise
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



#################### default 
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
default_params["svdsk"]        = 1000


##################### default params for representation 
rep_default_params                 = copy.copy(default_params)
rep_default_params["h"]            = 100
rep_default_params["ha"]           = act.tanh
rep_default_params["oa"]           = act.sgmoid

rep_default_params["l"]            = lo.negative_log_likelihood
rep_default_params["l2"]           = 0.001
rep_default_params["b"]            = 10
rep_default_params["i"]            = 20
rep_default_params["st"]           = st.instance_sampler
rep_default_params["sr"]           = 5 
rep_default_params["sp"]           = 0.01
rep_default_params["op"]           = op.gradient
rep_default_params["m"]            = m.internal_memory    
rep_default_params["r"]            = 0.1



##################################### default for leml
leml_default_params                 = copy.copy(default_params)

leml_default_params["h"]            = 100


leml_default_params["ha"]           = act.linear
leml_default_params["oa"]           = act.linear
leml_default_params["l"]            = lo.least_square
leml_default_params["l2"]           = 0.001
leml_default_params["b"]            = 10
leml_default_params["i"]            = 20
leml_default_params["op"]           = op.alternative_least_square
leml_default_params["m"]            = m.internal_memory
leml_default_params["svdsk"]        = 1000


#################################### default for wsabie
wsabie_default_params               = copy.copy(default_params)

wsabie_default_params["h"]          = 100
wsabie_default_params["l2"]         = 0.001
wsabie_default_params["b"]          = 10
wsabie_default_params["i"]          = 20
wsabie_default_params["sp"]         = 0.01
wsabie_default_params["op"]         = op.gradient
wsabie_default_params["m"]          = m.internal_memory
wsabie_default_params["r"]          = 0.1

wsabie_default_params["ha"]         = act.tanh
wsabie_default_params["oa"]         = act.linear
wsabie_default_params["l"]          = lo.weighted_approximate_rank_pairwise
wsabie_default_params["op"]         = op.gradient
wsabie_default_params["st"]         = st.full_sampler

