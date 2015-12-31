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
        self.argv.append("-l2");
        self.argv.append("0.01");
        self.argv.append("train_file");
        self.argv.append("m_file");
    def test_parseParameter(self):
        parameters = train.parseParameter(self.argv);

    def test_train(self):
        argv = ["train.py","-i","2","-b","2",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv) 

        
        argv = ["train.py","-i","2","-b","2",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv)

        
        argv = ["train.py","-i","2","-b","2",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv)

        argv = ["train.py","-i","2","-b","2","-st","0",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv)

    def test_train_al(self):
        argv = ["train.py", "-i","2","-b","2","-st","0",\
                "-l","1","-ha","1","-oa","1","-op","1",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv)

    def test_train_param(self): 
        argv = ["train.py","-ha","1","-oa","1","-l","1","-i","2","-b","2",\
                "tests/test_train_data.arff",\
                "tests/test_train_model.model"]
        train.main(argv)
