

def TransModelNonrigid(vertex_input, TransNonrigid):
    vertex = vertex_input.copy()
    for i in range(vertex.shape[1]):
        Trans = TransNonrigid[i*4:i*4+4,:].T
        vertex[:,i] = Trans[:, 0:3] @ vertex[:,i] + Trans[:,3]
    return vertex