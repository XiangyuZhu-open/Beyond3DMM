#!/usr/bin/env python3
# coding: utf-8

"""
Reference: https://github.com/YadiraF/PRNet/blob/master/utils/pose.py
"""

from math import cos, sin, atan2, asin, sqrt
import numpy as np



def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z


def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def rotate_vertex_img(vertex, angle, img_center):
    tx = img_center[0]
    ty = img_center[1]
    vertex[0, :] = vertex[0, :] - tx
    vertex[1, :] = vertex[1, :] - ty

    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    vertex = R @ vertex

    vertex[0, :] = vertex[0, :] + tx
    vertex[1, :] = vertex[1, :] + ty

    return vertex


def RotationMatrix(angle_x, angle_y, angle_z):
    phi = angle_x
    gamma = angle_y
    theta = angle_z

    R_x = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(gamma), 0, -np.sin(gamma)], [0, 1, 0], [np.sin(gamma), 0, np.cos(gamma)]])
    R_z = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    R = R_x @ R_y @ R_z

    return R





def main():
    pass


if __name__ == '__main__':
    main()
