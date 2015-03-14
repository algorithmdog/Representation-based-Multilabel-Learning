#!/bin/python
import os;
import sys;

path = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path);
sys.path.append(path + "/../utils/Python_Utils");
sys.path.append(path + "/../Python_Utils");

from Matrix_Utils  import *;
from latent_factor import *;
import unittest;

import pickle;

class LatentFactorTester(unittest.TestCase):
    def setUp(self):
        self.parameters = dict();
        self.parameters["num_feature"]   = 2;
        self.parameters["num_factor"]    = 2;
        self.parameters["num_label"]     = 2;
        self.parameters["sizes"]         = [3];
        self.parameters["hidden_active"] = "tanh";
        self.parameters["output_active"] = "sgmoid";
        self.parameters["loss"]          = "negative_log_likelihood";
        self.parameters["ins_lambda"]    = 0.001;
        self.parameters["label_lambda"]  = 0.001; 

    def init_model(self):
        model = Model(self.parameters);
        
        ##init w and b
        for idx in xrange(len(model.num) - 1):
            model.w[idx] = np.ones(model.w[idx].shape) / 100.0;        
            model.b[idx] = np.ones(model.b[idx].shape) / 100.0;
        
        ##init lw and lb;
        model.lb = np.ones(model.lb.shape) / 100.0;
        model.lw = np.ones(model.lw.shape) / 100.0;
        
        return model;

    
    def init_model1(self):
        model = Model(self.parameters);
        
        ##init w and b
        for idx in xrange(len(model.num) - 1):
            model.w[idx] = np.ones(model.w[idx].shape) / 10.0;        
            model.b[idx] = np.ones(model.b[idx].shape) / 10.0;
            model.w[idx][0,0] = 0;
        
        ##init lw and lb;
        model.lb = np.ones(model.lb.shape) / 10.0;
        model.lw = np.ones(model.lw.shape) / 10.0;
        model.lw[0,0] = 0;
        
        return model;
        
    

    def test_dimension(self):
        parameters = dict();
        parameters["num_feature"] = 100;
        parameters["num_label"]   = 100;
        model = Model(parameters);
        
        x = np.array([[0 for j in xrange(100)] for i in xrange(2)]);
        y = np.array([[0 for j in xrange(100)] for i in xrange(2)]);        
        self.assertTrue(model.check_dimension(x,y), True);

        x = np.array([[0 for j in xrange(101)] for i in xrange(2)]);
        with self.assertRaises(Exception):
            model.check_dimension(x);
    
        x = np.array([[0 for j in xrange(100)] for i in xrange(2)]);
        y = np.array([[0 for j in xrange(101)] for i in xrange(2)]);
        with self.assertRaises(Exception):
            model.check_dimension(x,y);


    def test_ff(self):
        model  = self.init_model();
        x      = np.array([[1,2],[0,1]]);
        expect = np.array([[0.5025559745374354, 0.5025559745374354],\
                           [0.502552977413684,  0.502552977413684]]);
        output = model.ff(x);
        self.assertTrue(is_matrix_equals(output, expect), True);
    
        #s = pickle.dumps(output);
        #f = open("output","w");
        #f.write(s);
        #f.close();

    def test_bp(self):
        model = self.init_model();
        x     = np.array([[1,2],[0,1]]);
        y     = np.array([[1,0],[0,1]]);
        idx   = np.array([[1,1],[1,1]]);
        model.bp(x, y, idx);

        grad_lb  = np.array([0.0051089519511194892, 0.0051089519511194892]);
        grad_lb /= 2;
        self.assertTrue(is_matrix_equals(grad_lb, model.grad_lb));

        grad_lw  = np.array([[-0.00024403,  0.00035541],[-0.00024403,  0.00035541]]);
        grad_lw /= 2;
        self.assertTrue(is_matrix_equals(grad_lw, model.grad_lw));

        grad_b1 = np.array([  2.55597454e-05,   2.55297741e-05]);
        self.assertTrue(is_matrix_equals(grad_b1, model.grad_b[1]));
        grad_w1 = np.array([[  7.66186099e-07,   7.66186099e-07],\
                            [  7.66186099e-07,   7.66186099e-07],\
                            [  7.66186099e-07,   7.66186099e-07]]);
        self.assertTrue(is_matrix_equals(grad_w1, model.grad_w[1]));


        grad_b0 = np.array([5.10895195e-07, 5.10895195e-07, 5.10895195e-07])
        self.assertTrue(is_matrix_equals(grad_b0, model.grad_b[0]));
        grad_w0 = np.array([[2.55597454e-07,  2.55597454e-07,  2.55597454e-07],\
                            [7.66492649e-07,  7.66492649e-07,  7.66492649e-07]])
        self.assertTrue(is_matrix_equals(grad_w0, model.grad_w[0]));
