# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 06:54:28 2017

@author: zhaoyafei
"""

import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank



class MatlabCp2tormException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
                __file__, super.__str__(self))

def tformfwd(trans, uv):

    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def tforminv(trans, uv):

    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy


def findNonreflectiveSimilarity(uv, xy, options=None):

    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print '--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print '--->X.shape: ', X.shape
    # print 'X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print '--->U.shape: ', U.shape
    # print 'U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    # print '--->r:\n', r

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    # print '--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print '--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def findSimilarity(uv, xy, options=None):

    options = {'K': 2}

#    uv = np.array(uv)
#    xy = np.array(xy)

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):


    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)

    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans):

    cv2_trans = trans[:, 0:2].T

    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):

    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)

    return cv2_trans

