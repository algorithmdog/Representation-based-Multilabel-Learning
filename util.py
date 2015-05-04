

def sparse_sum(sparse_matrix, axis = 0):
    m,n = sparse_matrix.shape
    if 0 == axis:
        result = [0.0 for r in xrange(n)]
        xy = sparse_matrix.nonzero()
        for i in xrange(len(xy[0])):
            x = xy[0][i]
            y = xy[1][i]
            result[y] += sparse_matrix[x,y]
        return result
    elif 1 == axis:
        result = [0.0 for r in xrange(m)]
        xy = sparse_matrix.nonzero()
        for i in xrange(len(xy[0])):
            x = xy[0][i]
            y = xy[1][i]
            result[x] += sparse_matrix[x,y]
        return result
    else:
        raise Exception("Invalid axis=%d"%axis)
