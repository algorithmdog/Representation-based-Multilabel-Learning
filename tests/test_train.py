#!/bin/python
import sys;
import os;
path  = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/utils/Python_Utils");
sys.path.append(path + "/../utils/Python_Utils");

from latent_factor import *;
from arffreader    import *;
import Logger;
import pickle;
import numpy as np;
import train;
import unittest;

class TrainTester(unittest.TestCase):
    def setUp(self):
        self.argv = [];
        self.argv.append("train");
        self.argv.append("-i");
        self.argv.append("0.01");
        self.argv.append("train_file");
        self.argv.append("m_file");
    def test_parseParameter(self):
        parameters = train.parseParameter(self.argv);
    def test_printUsages(self):
        train.printUsages();
