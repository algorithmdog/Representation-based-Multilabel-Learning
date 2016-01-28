#!/bin/python
import math
import sys

f = open(sys.argv[1])
data = []

for line in f:
    line = line.strip()
    if len(line) == 0: continue
    eles = line.split("|")
    eles = eles[0:len(eles)-1]
    
    d = [0 for i in xrange(len(eles))]
    for i in xrange(len(d)):
        k, v = eles[i].split(":")
        d[i] = float(v)
    data.append(d)

a1 = [0 for i in xrange(len(data[0]))]
for i in xrange(len(data)):
    for j in xrange(len(data[0])):
        a1[j] += data[i][j]
for j in xrange(len(a1)):
    a1[j] /= len(data);

print "average"
print a1

a2 = [0 for i in xrange(len(data[0]))]
for i in xrange(len(data)):
    for j in xrange(len(data[0])):
        a2[j] += (data[i][j] - a1[j])*(data[i][j] - a1[j])
for j in xrange(len(a2)):
    a2[j] = math.sqrt(a2[j] / (len(data) - 1)) 
        
print "standard deviation"
print a2     
