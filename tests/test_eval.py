#!/bin/python
import unittest
import eval
import numpy as np

class EvalTester(unittest.TestCase):
    def test_eval(self):
        p = np.array([[1,0],[1,0]])
        t = np.array([[1,1],[1,0],[1,0]])
        with self.assertRaises(Exception):
            eval.hamming(p,t)

        p = np.array([[1,0],[1,0],[0,0]])
        t = np.array([[1,1],[1,0],[0,1]])
 
        ham = eval.hamming(p,t)
        self.assertTrue(abs(ham-1.0/3) < 1e-6)
   
        ins_f = eval.instance_F(p,t)
        self.assertTrue(abs(ins_f-5.0/9) < 1e-6)
        
        label_f = eval.label_F(p,t)
        self.assertTrue(abs(label_f - 0.5) < 1e-6)

        p = np.array([[1,1,0],[0,1,1]])
        t = np.array([[1,0,1],[0,0,1]])
        label_f = eval.label_F(p,t)
        self.assertTrue(abs(label_f-5.0/9) < 1e-6)     
