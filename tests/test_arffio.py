#!/bin/bash
import arffio
import unittest
import numpy as np
import scipy as sc
import scipy.sparse as sp

class ArffioTest(unittest.TestCase):
	def test_sparse_read(self):
		i = 0;
