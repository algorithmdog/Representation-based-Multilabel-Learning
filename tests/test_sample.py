#!/bin/python
import os
import sys
import unittest
import numpy as np
import random

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/../utils/Python_Utils")
sys.path.append(path + "/../Python_Utils")

from Matrix_Utils import *
from sampler      import *


class SampleTester(unittest.TestCase):
    def test_get_sample(self):
        params = dict()
        params["sample_type"] = "full"
        sample = get_sample(params)
        self.assertEquals(sample, None)

        params["sample_type"] = "correlation_sample"
        sample = get_sample(params)
        self.assertTrue(isinstance(sample, CorrelationSampler));

        params["sample_type"] = "instance_sample"
        sample = get_sample(params)
        self.assertTrue(isinstance(sample, InstanceSampler))

        params["sample_type"] = "label_sample"
        params["num_label"] = 10
        sample = get_sample( params)
        self.assertTrue( isinstance(sample, LabelSampler) )

        with self.assertRaises(Exception):
            params["sample_type"] = "xxx" 
            get_sample(params)
    
    def test_sample(self):
        random.seed(0)

        params = dict()
        params["sample_type"] = "instance_sample"
        y = np.array([[1,0,0,0],[0,1.0,1,0]])
        ins_sample = get_sample(params)
        idx = ins_sample.sample(y)
        true = np.array([[1,0,0,1],[1,1,1,1]])
        self.assertTrue(is_matrix_equals(true, idx))    

        params = dict()
        params["sample_type"] = "label_sample"
        label_sample = get_sample( params)
        idx = label_sample.sample(y)
        true = np.array([[1,1,1,0],[1,1,1,0]])
        self.assertTrue(is_matrix_equals(true, idx))
        
        

