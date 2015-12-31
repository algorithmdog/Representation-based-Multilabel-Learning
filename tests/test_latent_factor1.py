#!/bin/python
import os;
import sys;
sys.path.append("./tests")

from common import *
from latent_factor import *;
import unittest;
import scipy.sparse as sp
import active 
import pickle;
import copy;

class ALLatentFactorTester(unittest.TestCase):
    def setUp(self):
        self.parameters = copy.deepcopy(default_params);
        self.parameters["h"]             = 1;
        self.parameters["nx"]            = 2;
        self.parameters["ny"]            = 2;
        self.parameters["sizes"]         = [];
        self.parameters["ha"]            = act.linear;
        self.parameters["oa"]            = act.linear;
        self.parameters["l"]             = lo.least_square;
        self.parameters["ins_lambda"]    = 0.001;
        self.parameters["label_lambda"]  = 0.001;
        self.parameters["l2"]            = 100; 
        

    def init_model(self):
        model = Model(self.parameters);
        
        model.w[0] = np.ones(model.w[0].shape) / 100.0
        model.b[0] = np.ones(model.b[0].shape) / 100.0 
        model.lb = np.ones(model.lb.shape) / 100.0;
        model.lw = np.ones(model.lw.shape) / 100.0;
        
        return model;

        
    def test_al(self):
        e = 0.000000001
        model = self.init_model();
        self.assertEquals(model.w[0].shape[0],2)
        self.assertEquals(model.w[0].shape[1],1)
        self.assertEquals(model.lw.shape[0],1)
        self.assertEquals(model.lw.shape[1],2)
        
        x = np.array([[1 for j in xrange(2)] for i in xrange(10)])
        y = np.array([[1 for j in xrange(2)] for i in xrange(10)]) 
        model.al_update(x,y,None)

        for i in xrange(len(model.b[0])):
            self.assertTrue(model.b[0][i] < e)
        for i in xrange(len(model.lb)):
            self.assertTrue(model.lb[i] < e)

        expected_lw = np.array([[.00199992, .00199992]])
        for i in xrange(model.lw.shape[0]):
            for j in xrange(model.lw.shape[1]):
                self.assertTrue(abs(expected_lw[i][j] - model.lw[i][j]) < e)
        
        expected_w  = np.array([[8 + 1.0/3],[8 + 1.0/3]])
        for i in xrange(model.w[0].shape[0]):
            for j in xrange(model.w[0].shape[1]):
                self.assertTrue(abs(expected_w[i][j] - model.w[0][i][j]) < e)


    def test_al1(self):
        e = 0.000000001
        model = self.init_model();
        
        x = sp.csr_matrix(np.array([[1 for j in xrange(2)] for i in xrange(10)]))
        y = sp.csr_matrix(np.array([[1 for j in xrange(2)] for i in xrange(10)])) 
        model.al_update(x,y,None)

        for i in xrange(len(model.b[0])):
            self.assertTrue(model.b[0][i] < e)
        for i in xrange(len(model.lb)):
            self.assertTrue(model.lb[i] < e) 

        expected_lw = np.array([[.00199992,.00199992]])
        for i in xrange(model.lw.shape[0]):
            for j in xrange(model.lw.shape[1]):
                self.assertTrue(abs(expected_lw[i][j] - model.lw[i][j]) < e)

        expected_w  = np.array([[8 + 1.0/3],[8 + 1.0/3]])
        for i in xrange(model.w[0].shape[0]):
            for j in xrange(model.w[0].shape[1]):
                self.assertTrue(abs(expected_w[i][j] - model.w[0][i][j]) < e)
