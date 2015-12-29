#!/bin/python

import os
import sys
import unittest
import numpy as np
import scipy.sparse as sp
import scipy as sc
import random

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/tests")

from Matrix_Utils import *
from sampler      import *
from sparse_sum   import *


class SampleTester(unittest.TestCase):
    def test_sum(self):
        a = sp.csr_matrix([[0,0,1],[1,0,1]])
        sum0 = sparse_sum(a,0)
        expected_sum0 = [1,0,2]
        for e1,e2 in zip(sum0,expected_sum0):
            self.assertEquals(e1,e2)

        sum1 = sparse_sum(a,1)
        expected_sum1 = [1,2]
        for e1,e2 in zip(sum1, expected_sum1):
            self.assertEquals(e1,e2)

        with self.assertRaises(Exception):
            sum2 = sparse_sum(a,2)

    def test_get_sampler(self):
        params = dict()
        params["st"] = st.full_sampler
        sample = get_sampler(params)
        self.assertTrue( isinstance(sample, FullSampler) )


        params["st"] = st.instance_sampler
        sample = get_sampler(params)
        self.assertTrue(isinstance(sample, InstanceSampler))


        with self.assertRaises(Exception):
            params["st"] = 110000 
            get_sampler(params)
    
    def test_sample(self):
        random.seed(0)
        
        to_do='''
        params = dict()
        params["sample_type"] = "instance_sample"
        y = np.array([[1,0,0,0],[0,1.0,1,0]])
        ins_sample = get_sampler(params)
        idx = ins_sample.sample(y)
        true = np.array([[1,0,0,1],[1,1,1,1]])
        self.assertTrue(is_matrix_equals(true, idx))    

        params = dict()
        params["sample_type"] = "label_sample"
        label_sample = get_sampler( params)
        idx = label_sample.sample(y)
        true = np.array([[1,1,1,0],[1,1,1,0]])
        self.assertTrue(is_matrix_equals(true, idx))
        '''
        

