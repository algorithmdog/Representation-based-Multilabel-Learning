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
import arff
import logging,Logger

label_flag = u'multi_label_';

class SvmWriter:
    def __init__(self, filename, num_feature, num_label):
        self.file = open(filename, "w")
        line = "#num_feature=%d num_label=%d\n"%(num_feature,num_label)
        print line
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

        x = sp.lil_matrix((num_ins, self.num_feature))
        y = sp.lil_matrix((num_ins, self.num_label))

        for i in xrange(len(lines)):
            line = lines[i]
            line = line.strip()
            eles = line.split(" ")       

            if ":" not in eles[0]:
                for j in xrange(1,len(eles)):
                    kv = eles[j].split(":")
                    x[i,int(kv[0])] = float(kv[1])
            
                labels = eles[0].strip().split(",")
                for j in xrange(len(labels)):
                    y[i,int(labels[j])] = 1

            else:
                for j in xrange(0,len(eles)):
                    kv = eles[j].split(":")
                    x[i,int(kv[0])] = float(kv[1])
        
        return sp.csr_matrix(x), sp.csr_matrix(y)
    
    def full_read(self):
        x, y, has_next = self.read()
        ##setting the larger batch
        m1,n1 = x.shape
        m2,n2 = y.shape
        num = n1 + n2

        while True == has_next:
            ix, iy, has_next = self.read()
            x = sp.vstack([x,ix], 'csr')
            y = sp.vstack([y,iy], 'csr')
        return x,y

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


class ArffWriter:
    def __init__(self, filename):
        self.encoder = arff.ArffEncoder();
        self.file    = open(filename, "w");

    def write(self, obj, is_first_call = True):
        gen = self.encoder.iter_encode(obj, is_first_call);
        for row in gen:
            self.file.write(row + u'\n');

    def close(self):
        self.file.close();
         

class ArffReader:

    def __init__(self, filename, batch = 50):
        self.arff_file     = open(filename, 'rb')
        self.decoder       = arff.ArffDecoder()
        self.batch         = batch
        self.num_class     = 0
        self.num_feature   = 0
        self.num_attribute = 0
        self.classes       = []
        self.first         = True
        #nextobj store the next obj. The nextobj is important
        #because arffreader should stop reading when nextobj 
        #has no data.  
        #It prevents [] from where the number of instances is 
        #a multiple of the size of batch.  
         
        self.nextobj = None


    def full_read_sparse(self):
        x, y, has_next = self.read_sparse()
        ##setting the larger batch
        m1,n1 = x.shape
        m2,n2 = y.shape
        num = n1 + n2

        while True == has_next:
            ix, iy, has_next = self.read_sparse()
            x = sp.vstack([x,ix], 'csr')
            y = sp.vstack([y,iy], 'csr')
        return x,y

    def full_read(self):
        x, y, has_next = self.read()
        while has_next:
            ix, iy, has_next = self.read()
            x = np.vstack([x,ix])
            y = np.vstack([y,iy])
        return x,y;

    def read_sparse(self):
        x, y, has_next = self.read()
        return sp.csr_matrix(x), sp.csr_matrix(y), has_next

    def read(self):
        x = None
        y = None

        if None != self.nextobj:
            data = self.nextobj["data"];
            obj  = dict();
            obj["data"] = data;
        
        self.nextobj = self.decoder.iter_decode(self.arff_file, \
                                                  obj = self.nextobj, \
                                                  batch = self.batch)

        if True == self.first:
            self.first = False;
            
            self.num_attribute = len( self.nextobj['attributes'] )
            self.classes = [ 0 for i in xrange(self.num_attribute) ]
            for i in xrange(self.num_attribute):
                feat = self.nextobj["attributes"][i][0];
                type = self.nextobj["attributes"][i][1];

                ## only support NUMERIC attributes
                if u'NUMERIC' != type:
                    logger = logging.getLogger(Logger.project_name);
                    logger.error("Only support NUMERIC attributes, "
                                 "but the %s feature is %s."%(feat,type));
                    raise Exception("Only support NUMERIC attributes, "
                                 "but the %s feature is %s."%(feat, type) );

                if label_flag in feat:
                    self.classes[i] = 1
                    self.num_class += 1


            self.num_feature = self.num_attribute - self.num_class
        
            data         = self.nextobj["data"];
            obj          = dict();
            obj["data"]  = data;
            self.nextobj = self.decoder.iter_decode(self.arff_file, \
                                                    obj   = self.nextobj, \
                                                    batch = self.batch);
 
        num_instance = len(obj['data'])
        x = [ [ 0 for col in xrange(self.num_feature) ] \
              for row in xrange(num_instance) ]
        y = [ [ 0 for col in xrange(self.num_class) ] \
              for row in xrange(num_instance) ]
        for i in xrange(num_instance):
            idx_x = 0
            idx_y = 0
            for j in xrange(self.num_attribute):
                if 1 == self.classes[j]:
                    y[i][idx_y] = obj['data'][i][j]
                    idx_y += 1
                else:
                    x[i][idx_x] = obj['data'][i][j]
                    idx_x += 1

        has_next = len(self.nextobj["data"]) != 0;
        return np.array(x), np.array(y), has_next;

    

    def close(self):
        self.arff_file.close()




