from sklearn.neighbors import NearestNeighbors
import numpy as np
import skimage
from sklearn.neighbors import KDTree

def nearest_neighbor_fitting(src, dst):
    tree = KDTree(dst)
    dist, ind = tree.query(src)
    return dist.ravel(), ind.ravel()



def nearest_neighbor(src, dst):
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    :param A: Nxm numpy array of corresponding points
    :param B: Nxm numpy array of corresponding points
    :return:
           T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
           R: mxm rotation matrix
           t: mx1 translation vector
    '''
    assert A.shape == B.shape

    m = A.shape[1]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

def icp(A, B,  max_iterations=20, tolerance=0.01, is_equal=False):
    #assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        if is_equal:
            distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
        else:
            distances, indices = nearest_neighbor_fitting(src[:m, :].T, dst[:m, :].T)


        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        #print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return distances, indices, mean_error, T