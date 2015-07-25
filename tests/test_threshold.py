#!/bin/python
import unittest
import threshold
import scipy.sparse as sp
import numpy as np

class thresholdTest(unittest.TestCase):
    def test_threshold(self):
        thr = threshold.ThresholdSel()
        p = np.array([[0,0.5],[-0.5,1]])
        y = sp.csr_matrix([[0,1],[1,0]])
        thr.update(p,y)
        self.assertEquals(thr.threshold, 0)
        self.assertEquals(thr.num,2)

        p = np.array([[0,0.5],[-0.5,1]])
        y = sp.csr_matrix([[1,1],[1,1]])
        thr.update(p,y)
        self.assertEquals(thr.threshold,(0+thr.low)/2.0)
        self.assertEquals(thr.num,4)

        p = np.array([[0.458,1]])
        y = sp.csr_matrix([[0,1]])
        thr.update(p,y)
        self.assertEquals(thr.threshold, ((0+thr.low) /2.0 * 4 + 0.46)/5 )
        self.assertEquals(thr.num, 5)
