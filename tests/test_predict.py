#!/bin/python
import sys;
import os;
path  = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/utils/Python_Utils");
sys.path.append(path + "/../utils/Python_Utils");

from latent_factor import *;
from arffio        import *;
import Logger;
import pickle;
import numpy as np;
import predict;
import unittest;

class PredictTester(unittest.TestCase):
    def setUp(self):
        self.argv = [];
        self.argv.append("predict.py");
        self.argv.append("test_file");
        self.argv.append("res_file");
        self.argv.append("m_file");
    def test_parseParameter(self):
        parameters = predict.parseParameter(self.argv);
    def test_printUsages(self):
        predict.printUsages();
