#!/bin/python
import math
import numpy as np

class ThresholdSel:
    def __init__(self):
        self.threshold = 0
        self.num       = 0
        self.low       = -5
        self.high      = 5
        self.step      = 0.01
    def update(self, predict, gold):
        #print 'predict'
        #print predict
        #print 'gold'
        #print gold.todense()
        m,n  = gold.shape
        step = self.step
        low  = self.low
        high = self.high
        gold_lcard  = self.lcard_sparse(gold);

        #print predict
        numSplit = int(math.ceil( (high-low) / step))
        distributes = np.zeros(numSplit + 1) #[0.0 for i in xrange(numSplit+1)]
        predict = np.ceil((predict-low)/step)
        predict = predict.astype(np.int32)
        predict[predict < 0] = 0
        predict[predict > numSplit] = numSplit

        bincount = np.bincount(predict.reshape(-1),minlength=numSplit+1)
        distributes += bincount
        #for i in xrange(m):
        #    for j in xrange(n):
        #        idx = predict[i,j]
        #        distributes[idx] += 1.0;        

        
        threshold = low
        sum1 = distributes[0] 
        mindiff   = abs(n - sum1/m - gold_lcard)
        #print 'threshold',low,' lcard',n-sum1/m, ' glcard',gold_lcard
        for i in xrange(1,numSplit+1):
            sum1 += distributes[i]
            lcard = n - sum1 / m
        #   print 'threhold:',low+i*step,' lcard', lcard, ' glcard',gold_lcard
            if mindiff > abs(lcard - gold_lcard):
                threshold =  low + i * step
                mindiff = abs(lcard - gold_lcard)
        #print "thresholdsel,",threshold

        self.threshold = (self.threshold * self.num + threshold * m)/(self.num + m) 
        self.num       = self.num + m

    def lcard_sparse(self,y):
        m,n = y.shape
        return len(y.nonzero()[0])* 1.0  / m;
       
