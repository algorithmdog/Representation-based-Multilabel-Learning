#!/bin/python
import logging;
import sys;

project_name = "multilabel";

instance = logging.getLogger(project_name);
instance.setLevel(logging.INFO);

handler = logging.StreamHandler(sys.stderr);
handler.setLevel(logging.INFO);

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s");
handler.setFormatter(formatter);

instance.addHandler(handler);

def initlog(opts):
    global instance;
    global handler;
    global project_name;

    print opts;
    if "project_name" in opts:
        project_name = opts["project_name"];
        print "in Logger", project_name;

    instance.removeHandler(handler);
    instance = logging.getLogger(project_name);

    #set longer
    if "logfile" in opts:
        handler = logging.FileHandler(opts["logfile"]);
    else:
        handler = logging.StreamHandler(sys.stderr);
    
    #set formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s");
    handler.setFormatter(formatter);

    ##set level
    instance.setLevel(logging.INFO);
    if "level" in opts:
        if "notset" == opts["level"].lowcase():
            instance.setLevel(logging.NOTSET)
        elif "debug" == opts["level"].lowcase():
            instance.setLevel(logging.DEBUG)
        elif "info"  == opts["level"].lowcase():
            instance.setLevel(logging.INFO)
        elif "warning" == opts["level"].lowcase():
            instance.setLevel(logging.WARNING)
        elif "error" == opts["level"].lowcase():
            instance.setLevel(logging.ERROR)
        elif "critical" == opts["level"].lowcase():
            instance.setLevel(logging.critical)   

    instance.addHandler(handler);
   
