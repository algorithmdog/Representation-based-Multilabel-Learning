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

    def test_train(self):
        argv = ["train.py","-b","2","-t","full",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv) 

        
        argv = ["train.py","-b","2","-t","instance_sample",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv)

        
        argv = ["train.py","-b","2","-t","label_sample",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv)
