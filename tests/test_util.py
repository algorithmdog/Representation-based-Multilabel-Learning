#!/bin/python
import os
import sys
import unittest
import numpy as np
import scipy.sparse as sp
import scipy as sc

import random

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + "/../utils/Python_Utils")
sys.path.append(path + "/../Python_Utils")

from Matrix_Utils import *
from util      import *


class utilTester(unittest.TestCase):
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

        
        

