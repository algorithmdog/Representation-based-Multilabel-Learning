#!/bin/python
import os;
import sys;
import unittest;
import arffio;

path = os.path.split(os.path.realpath(__file__))[0];

class test_reader(unittest.TestCase):

    def equals(self, x, y):
        mx = len(x);
        nx = len(x[0]);
        my = len(y);
        ny = len(y[0]);
        if mx != my or nx != ny:
            return False;

        for i in xrange(mx):
            for j in xrange(nx):
                if abs(x[i][j] - y[i][j]) > 1e-6:
                    return False;

        return True;

        
