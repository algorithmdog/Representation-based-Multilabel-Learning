#!/bin/python
import arffio
import unittest

class testsvmreader(unittest.TestCase):
    def testsvmread(self):
        svmreader = arffio.SvmReader("./tests/test_svmread_data", batch = 2)
        x, y, has_next = svmreader.read() 
        self.assertEquals(has_next, True)       
        self.assertEquals(x[0,1],4.5)
        self.assertEquals(x[1,19],1)
        self.assertEquals(y[0,1],1)
        x, y, has_next = svmreader.read()
        self.assertEquals(has_next, False)
        self.assertEquals(y[0,1],1)
        self.assertEquals(y[0,5],1)
        self.assertEquals(x[0,10],2)
        self.assertEquals(x[0,0],0)
        svmreader.close()

        svmreader = arffio.SvmReader("./tests/test_svmread_data", batch = 50)
        x, y, has_next = svmreader.read()
        
