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


#ha = Enum(["tanh","linear"]);
#oa = Enum(["sgmoid","linear"]);
op = Enum(["gradient","alternative_least_square"]);
st = Enum(["full_sampler","instance_sampler"]);
