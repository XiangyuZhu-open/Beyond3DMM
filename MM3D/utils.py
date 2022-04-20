import numpy as np
from .cython import mesh_core_cython
import os

def NormDirection(vertex, tri):
    vertex_normal = np.zeros((vertex.shape[0], 3), dtype=np.float32)

    vertex = vertex.astype(np.float32).copy()
    tri = tri.astype(np.int32).copy()
    mesh_core_cython.get_normal(vertex_normal, vertex, tri, vertex.shape[0], tri.shape[0])

    return vertex_normal
def get_all_files(dir, suffix):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            if(path.endswith(suffix)):
                path = path[len(dir):]
                files_.append(path)
    return files_





