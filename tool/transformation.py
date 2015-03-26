#!/bin/python
import sys;
import os;
path = os.path.split(os.path.realpath(__file__))[0];
sys.path.append(path + "/../utils/liac-arff");
sys.path.append(path + "/../../utils/liac-arff");
sys.path.append(path + "/..");

import arff;
import arffio;
default_minbatch = 200;

def printUsages():
    print "Usage: transformation.py origin_arff_file flag target_arff_file"; 

def parseParameters(argv):
    if len(argv) < 4: # at least 4 parameters
        printUsages();
        exit()

    parameters                = dict();
    parameters["origin_file"] = argv[len(argv) - 3];
    parameters["flag"]        = argv[len(argv) - 2];
    parameters["target_file"] = argv[len(argv) - 1];
    
    return parameters;

def transformation(parameters, minbatch = default_minbatch):
    origin_file = parameters["origin_file"];
    flag        = parameters["flag"];
    target_file = parameters["target_file"];

    rf = open(origin_file);
    wf = open(target_file, "w");

    first   = True;
    transferredObj = None;
    generator = None

    decoder = arff.ArffDecoder();
    encoder = arff.ArffEncoder();    
    obj     = decoder.iter_decode(rf, obj = None, batch = minbatch);

    while True:
        if True == first:
            first = False;
            transferredObj  = obj;    
            for i in xrange(len(transferredObj["attributes"])):
                name = u'%s'%transferredObj["attributes"][i][0];
                type = u'NUMERIC';
                if u'%s'%flag in name:
                    name = u'%s'%arffio.label_flag + name;
                transferredObj["attributes"][i] = (name, type);

            generator = encoder.iter_encode(transferredObj, \
                                            is_first_call = True);
        
        else:
            transferredObj = dict();
            transferredObj["data"] = obj["data"];
            generator = encoder.iter_encode(transferredObj, \
                                            is_first_call = False);
  
        # write the encoded data to target_file
        for row in generator:
             wf.write(row + u'\n');

        obj = decoder.iter_decode(rf, obj = obj, batch = minbatch);
        if 0 == len(obj["data"]):   break; 
   
    rf.close();
    wf.close();

if __name__ == "__main__":
    parameters = parseParameters(sys.argv); 
    transformation(parameters);
