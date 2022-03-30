#!/usr/bin/env python3
# coding: utf-8
__author__ = 'cleardusk'

import sys
sys.path.append('../../')

import numpy as np
from math import sqrt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .ddfa import reconstruct_vertex
from MM3D.params import *


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.float32)
    else:
        res = np.zeros((dh, dw), dtype=np.float32)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    if (sy < 0 or ey < 0 or sx < 0 or ex < 0):
        return None

    if (sy > h or ey > h or sx > w or ex > w):
        return None


    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box

def parse_roi_box_from_landmark_extend(pts, extend_scale):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) * extend_scale
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    return roi_box


def dump_to_ply(vertex, tri, wfp):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""

    n_vertex = vertex.shape[1]
    n_face = tri.shape[1]
    header = header.format(n_vertex, n_face)

    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_vertex):
            x, y, z = vertex[:, i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        for i in range(n_face):
            idx1, idx2, idx3 = tri[:, i]
            f.write('3 {} {} {}\n'.format(idx1 - 1, idx2 - 1, idx3 - 1))
    print('Dump tp {}'.format(wfp))


def dump_vertex(vertex, wfp):
    sio.savemat(wfp, {'vertex': vertex})
    print('Dump to {}'.format(wfp))


def dump_paras(R, t3d, alpha_shape, alpha_exp, wfp):
    sio.savemat(wfp, {'R': R, 't3d':t3d, 'alpha_shape':alpha_shape, 'alpha_exp':alpha_exp})
    print('Dump to {}'.format(wfp))

def _predict_vertices(param, roi_bbox, dense, transform=True):
    vertex = reconstruct_vertex(param, dense=dense)
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex


def predict_68pts(param, roi_box):
    return _predict_vertices(param, roi_box, dense=False)


def predict_dense(param, roi_box):
    return _predict_vertices(param, roi_box, dense=True)


#def draw_landmarks(img, pts, rect, style='fancy', wfp=None, show_flg=False, **kwargs):
def draw_landmarks(img, pts, style='fancy', wfp=None, show_flg=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    #bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
    #plt.gca().add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1]))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')



    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if style == 'simple':
            plt.plot(pts[i][0, :], pts[i][1, :], 'o', markersize=4, color='g')

        elif style == 'fancy':
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color,
                         markeredgecolor=markeredgecolor, alpha=alpha)

    if wfp is not None:
        plt.savefig(wfp, dpi=200)
        print('Save visualization result to {}'.format(wfp))
    if show_flg:
        plt.show()


def get_colors(image, vertices):
    [h, w, _] = image.shape
    vertices[0, :] = np.minimum(np.maximum(vertices[0, :], 0), w - 1)  # x
    vertices[1, :] = np.minimum(np.maximum(vertices[1, :], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[1, :], ind[0, :], :]  # n x 3

    return colors


def write_obj_with_colors(obj_name, vertices, triangles, colors):
    triangles = triangles.copy() # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f} {} {} {}\n'.format(vertices[1, i], vertices[0, i], vertices[2, i], colors[i, 2],
                                               colors[i, 1], colors[i, 0])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[0, i], triangles[1, i], triangles[2, i])
            f.write(s)

def predict_3DMM_paras(param, roi_box, img_size):
    std_size = 120
    [height, width, nChannels] = img_size

    if len(param) == 62:
        param = param * param_std + param_mean
    else:
        param = np.concatenate((param[:11], [0], param[11:]))
        param = param * param_std + param_mean

    p_ = param[:12].reshape(3, -1)
    R = p_[:, :3]
    t3d = p_[:, -1].reshape(3, 1)
    alpha_shape = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    sx, sy, ex, ey = roi_box

    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    s = (scale_x + scale_y) / 2

    R =  s * R
    t3d[0] = s * t3d[0] + sx
    t3d[1] = s * t3d[1] - s * std_size - sy + height
    t3d[2] = s * t3d[2] + 0

    return R, t3d, alpha_shape, alpha_exp

def main():
    pass


if __name__ == '__main__':
    main()
