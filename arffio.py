# 2015.03.10 14:42:13 UTC
import sys
import os
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path + '/utils/Python_Utils')
sys.path.append(path + '../utils/Python_Utils')
sys.path.append(path + '/utils/liac-arff')
sys.path.append(path + '../utils/liac-arff')
import numpy as np
import scipy as sc
import scipy.sparse as sp
import logging,Logger

label_flag = u'multi_label_';

class SvmWriter:
    def __init__(self, filename, num_feature, num_label):
        self.file = open(filename, "w")
        line = "#num_feature=%d num_label=%d\n"%(num_feature,num_label)
        self.file.write(line)
     
    def write(self, x, y):
        m,n = x.shape
        labels   = [[] for r in xrange(m)]
        features = [[] for r in xrange(m)]
        ij = x.nonzero()
        for k in xrange(len(ij[0])):
            i = ij[0][k]
            j = ij[1][k]
            features[i].append("%d:%f"%(j,x[i,j]))
        ij = y.nonzero()
        for k in xrange(len(ij[0])):
            i = ij[0][k]
            j = ij[1][k]
            labels[i].append("%d"%j)
        
        for i in xrange(m):
            #print features[i]
            line = ",".join(labels[i]) + " " + " ".join(features[i]) + "\n"
            #print line
            self.file.write(line)

    def close(self):
        self.file.close()

class SvmReader:
    def __init__(self, filename, batch = 50):
        self.file          = open(filename)
        self.batch         = batch
        self.num_label     = 0
        self.num_feature   = 0
        self.next_x        = None
        self.next_y        = None
        
        ##read the comment line
        ##  the comment line should give num_feature and num_labels
        ##  for example '#num_feature=6\tnum_label=10'
        line = self.file.readline()
        line = line.strip()
        line = line.replace("#", "")
        eles = line.split(" ")
        #print "eles",eles
        #print "eles[0].split('=')",eles[0].split("=")
        #print "int((eles[0].split('='))[0])", int((eles[0].split("="))[1])
        self.num_feature = int((eles[0].split("="))[1])
        self.num_label   = int((eles[1].split("="))[1])        

    def parse(self,lines):
        num_ins = len(lines)
        if num_ins == 0:
            return None, None        

        #x = sp.lil_matrix((num_ins, self.num_feature))
        #y = sp.lil_matrix((num_ins, self.num_label))
        xr = []
        xc = []
        xd = []
        yr = []
        yc = []
        yd = []

        for i in xrange(len(lines)):
            line = lines[i]
            line = line.strip()
            eles = line.split(" ")       

            if ":" not in eles[0]:
                for j in xrange(1,len(eles)):
                    kv = eles[j].split(":")
                    #x[i,int(kv[0])] = float(kv[1])
                    xr.append(i)
                    xc.append(int(kv[0]))
                    xd.append(float(kv[1]))
                labels = eles[0].strip().split(",")
                #print "xxx",line,labels
                for j in xrange(len(labels)):
                    #y[i,int(labels[j])] = 1
                    yr.append(i)
                    yc.append(int(labels[j]))
                    yd.append(1)
            else:
                for j in xrange(0,len(eles)):
                    kv = eles[j].split(":")
                    #x[i,int(kv[0])] = float(kv[1])
                    xr.append(i)
                    xc.append(int(kv[0]))
                    xd.append(float(kv[1]))
        
        xi =  sp.csr_matrix((xd,(xr,xc)),(num_ins,self.num_feature))
        yi = sp.csr_matrix((yd,(yr,yc)),(num_ins,self.num_label))
    
        return xi, yi

    def full_read(self):
        lines = []
        for line in self.file:
            if line is None or len(line.strip()) == 0:  break
            #print "full_read",line
            lines.append(line.strip())
        
        return self.parse(lines)

    def read(self):
        if None == self.next_x:
            lines = []
            for i in xrange(self.batch):
                line = self.file.readline()
                if line is None or len(line.strip()) == 0:    break
                lines.append(line)

            self.next_x, self.next_y = self.parse(lines)                

        x = self.next_x
        y = self.next_y

        lines = []
        for i in xrange(self.batch):
            line = self.file.readline()
            if line is None or len(line.strip()) == 0:    break
            lines.append(line)
        self.next_x, self.next_y = self.parse(lines)                
        has_next = not (self.next_x is None);
        
        return x, y, has_next;
    
    
    def close(self):
        self.file.close()
