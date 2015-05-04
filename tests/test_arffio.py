#!/bin/bash
import arffio
import unittest
import numpy as np
import scipy as sc
import scipy.sparse as sp

class ArffioTest(unittest.TestCase):
    def test_sparse_read(self):
        data_file = "./tests/test_arffio_data.arff"
        reader = arffio.ArffReader(data_file, 2)
        x, y = reader.full_read_sparse()
        actual_x = x.todense()
        actual_y = y.todense()
        expected_x = np.array([[1.0,0,1],[0,1,0],[0,0,1],[1,0,0],[0,0,1]])
        expected_y = np.array([[1.0,0,0],[0,1,0],[1,0,0],[0,0,1],[1,0,0]])
        

        m,n = actual_x.shape
        for i in xrange(m):
            for j in xrange(n):
                self.assertEqual(actual_x[i,j], expected_x[i,j])

        m,n = actual_y.shape
        for i in xrange(m):
            for j in xrange(n):
                self.assertEqual(actual_y[i,j], expected_y[i,j])

