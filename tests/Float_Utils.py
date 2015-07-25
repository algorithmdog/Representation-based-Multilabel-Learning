#/bin/python
def eq(f1,f2):
    return abs(f1-f2) < 1e-6;
def gt(f1,f2):
    return f1-f2 > 1e-6;
def lt(f1,f2):
    return f2-f1 > 1e-6;    

