#!/bin/python
import os;
import sys;
sys.path.append("./tests")

from Matrix_Utils  import *;
from latent_factor import *;
import unittest;
import scipy.sparse as sp
import active 
import pickle;


class ALLatentFactorTester(unittest.TestCase):
    def setUp(self):
        self.parameters = dict();
        self.parameters["h"]             = 2;
        self.parameters["nx"]            = 2;
        self.parameters["ny"]            = 2;
        self.parameters["sizes"]         = [];
        self.parameters["ha"]            = act.linear;
        self.parameters["oa"]            = act.linear;
        self.parameters["l"]             = lo.least_square;
        self.parameters["ins_lambda"]    = 0.001;
        self.parameters["label_lambda"]  = 0.001;
        self.parameters["l2"]            = 0.001; 

    def init_model(self):
        model = Model(self.parameters);
        
        model.w[0] = np.ones(model.w[0].shape) / 100.0
        model.b[0] = np.ones(model.b[0].shape) / 100.0 
        model.lb = np.ones(model.lb.shape) / 100.0;
        model.lw = np.ones(model.lw.shape) / 100.0;
        
        return model;

        
    def test_al(self):
        return
