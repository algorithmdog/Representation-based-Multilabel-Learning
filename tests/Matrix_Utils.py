#/bin/python
from numpy import *;

import types
import Float_Utils;

def is_matrix_equals(M1,M2,epsilon = 1e-9):
    if 2 == M1.ndim: 
        if M1.shape != M2.shape:
            return False;
        r,c = M1.shape;
        for i in xrange(r):
            for j in xrange(c):
                if abs(M1[i,j] - M2[i,j]) >= epsilon:
                    return False;
        return True;
    elif 1 == M1.ndim:
        if M1.shape != M2.shape:
            return False;
        r = M1.shape[0];
        for i in xrange(r):
            if abs(M1[i] - M2[i]) >= epsilon:
                return False;
        return True;

    else:
        raise Exception("equals_matrix function not support ndim = %d"%(M1.ndim));

def matrix_pinv(A): # return pinv(A)
    m,n     = A.shape;
    u,d1,vt = linalg.svd(A);
    d       = zeros([m,n]);
    for i in xrange(min(m,n)):
        if not Float_Utils.eq(d1[i],0.0):
            d[i,i] = 1.0/d1[i];

    PINV = dot(dot(transpose(vt), transpose(d)), transpose(u));
    return PINV

def matrix_A_dot_PI(A,PI): #return A * PI. PI is the permutation matrix"
    if 1 != PI.ndim:
        raise Exception("matrix_A_dot_PI requires PI.ndim = 1, but PI.ndim=%d"%PI.ndim);

    m,n = A.shape;
    k   = len(PI);
    API = zeros([m,k]);

    for i in xrange(k):
        c            = PI[i];
        API[:,i:i+1] = copy(A[:,c:c+1]);

    return API;
  

def matrix_Ak(A,k): # A_k is the best rank-k approximination to A
    m,n = A.shape;
    if k > min(m,n):
        raise Exception("matrix_Ak requires k <= min(m,n), but m=%d, n=%d, k=%d"%(m,n,k));

    u,d,vt = linalg.svd(A);
    
    Ak = zeros([m,n]);
    for i in xrange(k):
        Ak += dot(u[:,i:i+1],vt[i:i+1,:]) * d[i];
    return Ak;

def matrix_read(filename):
    row = 0;
    col = -1;
    for line in file(filename, "r"):    
        row += 1;
        if -1 == col:
            line = line.strip();
            eles = line.split("\t");
            col  = len(eles);

    m = zeros([row,col]);
    i = 0;
    j = 0;
    for line in file(filename, "r"):
        line = line.strip();
        eles = line.split("\t");
        for j in xrange(len(eles)):
            m[i,j] = float(eles[j]);        
        i += 1;    

    return m;

def matrix_show(M, length=9):
    format = ' %.'+str(length)+'f\t'
    if 2 == M.ndim:
        r,c = M.shape;
        for i in xrange(r):
            for j in xrange(c):
                if (i > 3 and i < r-3) or (j > 3 and j < c - 3):
                    continue;
                elif i == 3 and i < r -3:
                    print "    ...\t\t",;
                elif j == 3 and j < c -3:
                    print "    ...\t\t",;   
                elif M[i,j] >= 0:
                    print format%M[i,j],
                else:
                    print format.strip()%M[i,j],
            if i>3 and i < r - 3:
                continue;
            else:
                print "";
    elif 1 == M.ndim:
        l = len(M)
        for i in xrange(l):
            if i > 3 and i < l-3:
                continue;
            elif i == 3 and i < l-3:
                print "    ...\t\t",
            elif M[i] >= 0:
                print format%M[i],
            else:
                print format%M[i],
        print "";
    else:
        raise Exception("show_matrix not support ndim=%d yet"%M.ndim);

