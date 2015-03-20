#!/bin/python
import os;
import sys;
path = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/../tool");

import unittest;
import transformation;

class TransformationTester(unittest.TestCase):

    def test_transformation(self):
        params = dict();
        params["origin_file"] = "./tests/test_transformation_data.arff";
        params["flag"]        = "label"
        params["target_file"] = "./tests/test_transformation_result.arff";
        transformation.transformation(params, 2);
        
