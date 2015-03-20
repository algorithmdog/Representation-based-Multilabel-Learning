#!/bin/python
import unittest;
import eval
import numpy as np;

class EvalTester(unittest.TestCase):
    def test_hamming(self):
        p = np.array([[1,0],[1,0]]);
        t = np.array([[1,1],[1,0],[1,0]]);
        with self.assertRaises(Exception):
            eval.hamming(p,t)

        t = np.array([[1,1],[1,0]]);
        ham = eval.hamming(p,t);
        self.assertTrue(abs(ham-0.25) < 1e-6);
    
