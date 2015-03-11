# 2015.03.10 14:42:13 UTC
import sys
import os
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + '/utils/Python_Utils')
sys.path.append(path + '../utils/Python_Utils')
sys.path.append(path + '/utils/liac-arff')
sys.path.append(path + '../utils/liac-arff')
import numpy as np
import arff

class ArffReader:

    def __init__(self, filename, batch = 50):
        self.arff = open(filename, 'rb')
        self.decoder = arff.ArffDecoder()
        self.batch = batch
        self.obj = None
        self.num_class = 0
        self.num_feature = 0
        self.num_attribute = 0
        self.classes = []



    def read(self):
        x = None
        y = None
        self.obj = self.decoder.iter_decode(self.arff, obj=self.obj, batch=self.batch)
#       print self.obj['data']
        if [] == self.classes:
            self.num_attribute = len(self.obj['attributes'])
            self.classes = [ 0 for i in xrange(self.num_attribute) ]
            for i in xrange(self.num_attribute):
                if u'multi_label_' in self.obj['attributes'][i][0]:
                    self.classes[i] = 1
                    self.num_class += 1

            self.num_feature = self.num_attribute - self.num_class
        num_instance = len(self.obj['data'])
        x = [ [ 0 for col in xrange(self.num_feature) ] for row in xrange(num_instance) ]
        y = [ [ 0 for col in xrange(self.num_class) ] for row in xrange(num_instance) ]
        for i in xrange(num_instance):
            idx_x = 0
            idx_y = 0
            for j in xrange(self.num_attribute):
                if 1 == self.classes[j]:
                    y[i][idx_y] = self.obj['data'][i][j]
                    idx_y += 1
                else:
                    x[i][idx_x] = self.obj['data'][i][j]
                    idx_x += 1


        return (np.array(x), np.array(y))



    def close():
        self.arff.close()




