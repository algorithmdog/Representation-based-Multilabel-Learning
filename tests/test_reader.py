#!/bin/python
import os;
import sys;
import unittest;
import arffreader;

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


    def test_read(self):
        reader = arffreader.ArffReader(path + "/test_reader_data.arff", 2);
        x,y = reader.read();
        tx = [[1,0, 1],[0, 1, 0]];
        ty = [[1,0, 0],[0, 1, 0]];
        self.assertEqual(self.equals(x, tx), True);
        self.assertEqual(self.equals(y, ty), True);

        x,y = reader.read();
        tx = [[0, 0, 1],[1, 0, 0]];
        ty = [[1, 0, 0],[0, 0, 1]];
        self.assertEqual(self.equals(x, tx), True);
        self.assertEqual(self.equals(y, ty), True);

        x,y = reader.read();
        tx = [[ 0, 0, 1]];
        ty = [[ 1, 0, 0]];
        self.assertEqual(self.equals(x, tx), True);
        self.assertEqual(self.equals(y, ty), True);

        x,y = reader.read();
        
