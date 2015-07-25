import numpy as np
import scipy.sparse as sp

def sparse_sum(sparse_matrix1, axis = 0):
    if not sp.isspmatrix_csr(sparse_matrix1):
        sparse_matrix = sp.csr_matrix(sparse_matrix1)
    else:
        sparse_matrix = sparse_matrix1
    m,n = sparse_matrix.shape

    if 0 == axis:
        result = np.zeros(n, np.int32)
        xy = sparse_matrix.nonzero()
        for i,j,v in zip(xy[0],xy[1],np.asarray(sparse_matrix.data)):
            result[j] += int(v)
        return result
    elif 1 == axis:
        result = np.zeros(m, np.int32)
        xy = sparse_matrix.nonzero()
        for i,j,v in zip(xy[0],xy[1],np.asarray(sparse_matrix.data)):
            result[i] += int(v)
        return result
    else:
        raise Exception("Invalid axis=%d"%axis)
