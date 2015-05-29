#!/bin/python
import math

class ThresholdSel:
    def __init__(self):
        self.threshold = 0
        self.num       = 0
        self.low       = -2
        self.high      = 2
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

        numSplit = int(math.ceil( (high-low) / step))
        distributes = [0.0 for i in xrange(numSplit+1)]
        for i in xrange(m):
            for j in xrange(n):
                idx = int(math.ceil((predict[i,j]-low) / step))
                if idx < 0: idx = 0
                if idx > numSplit: idx = numSplit
                distributes[idx] += 1.0;        

        
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
       
