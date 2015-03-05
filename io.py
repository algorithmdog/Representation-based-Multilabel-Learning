#!/bin/python
import numpy as np;
class MatrixReader:
    def __init__(self, x_filename, batch = 50, separater = " "):
        self.x_file    = open(x_filename);
        self.bath      = batch;
        self.separater = separater;
    def read(self);
        x = [];
        for i in xrange(batch):
            x_line = self.x_file.readline();
            if None == x_line or "" = x_line.strip():
                self.x_file.close();
                return False, np.array(x);

            eles = x_line.strip().split(separater);
            x.append(map(float,eles));

        has_next = self.x_file.has_next();
        self.x_file.close();
        return has_next, np.array(x);  


