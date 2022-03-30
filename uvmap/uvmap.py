import os
import numpy as np
import cv2
import dlib
from ..rend import render
from skimage import io
import scipy.io as sio
from scipy import interpolate
from .. import transform as trans3D


def file_name(file_dir):
    l=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.npy':
                l.append(os.path.join(root, file))
    return l


def idx_eye_corner(path_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../resource/shape_predictor_68_face_landmarks.dat')

    # cv2读取图像
    img = cv2.imread(path_image)

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    eye_corner = {}
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            # print(idx,pos)

            # # 利用cv2.circle给每个特征点画一个圈，共68个
            # cv2.circle(img, pos, 5, color=(0, 255, 0))
            # # 利用cv2.putText输出1-68
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)

            if idx == 39:
                eye_corner['39'] = pos
            if idx == 42:
                eye_corner['42'] = pos

    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    if '39' not in list(eye_corner.keys()) or '42' not in list(eye_corner.keys()):
        return 'drop'
    else:
        return eye_corner


def eye_dist(path_image):
    eye_corner = idx_eye_corner(path_image)
    if eye_corner == 'drop':
        return 'drop'
    else:
        (x_39, y_39) = eye_corner['39']
        (x_42, y_42) = eye_corner['42']
        dist = ((x_39 - x_42) ** 2 + (y_39 - y_42) ** 2) ** 0.5
        return dist


def pixel_dist(mat1, mat2, eyedist):
    (height, width, channel) = mat1.shape
    dist_mat = np.zeros((height, width))

    for h in range(height):
        for w in range(width):
            dist = np.linalg.norm(mat1[h, w] - mat2[h, w])
            dist_mat[h, w] = dist / eyedist

    return dist_mat


def mat_shift(mat_c, mat_f):
    a, b = np.where(mat_f[:, :, 2] == np.max(mat_f[:, :, 2]))

    x_f, y_f, z_f = mat_f[a, b].flatten()
    x_c, y_c, z_c = mat_c[a, b].flatten()

    x_shift = x_f - x_c
    y_shift = y_f - y_c

    mat_c[:, :, 0] += x_shift
    mat_c[:, :, 1] += y_shift

    return mat_c


def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:, 0] = uv_coords[:, 0]*(uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1]*(uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords


def generate_uv_g_t(transformed_vertices, colors, triangles, uv_coords):
    uv_h = uv_w = 256
    image_h = image_w = 256
    uv_coords = process_uv(uv_coords, uv_h, uv_w)

    #-- 1. uv texture map
    # attribute = colors
    # uv_texture_map = mesh.render.render_colors(uv_coords, triangles, attribute, uv_h, uv_w, c=3)
    # io.imsave('{}/uv_texture_map.jpg'.format(save_folder), np.squeeze(uv_texture_map))

    #-- 2. uv position map in 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network'
    #--   for face reconstruction & alginment(dense correspondences)
    # To some extent, when uv space is regular, position map is a subclass of geometry image(recording geometry information in regular image)
    # Notice: position map doesn't exit alone, it depends on the corresponding rendering(2d facical image).
    # Attribute is the position with respect to image coords system.
    # use standard camera & orth projection here
    projected_vertices = transformed_vertices.copy()
    image_vertices = trans3D.to_image(
        projected_vertices, image_h, image_w)
    position = image_vertices.copy()
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
    attribute = position
    # corresponding 2d facial image
    # image = mesh.render.render_colors(image_vertices, triangles, colors, image_h, image_w, c=3)
    uv_position_map = render.render_colors(
        uv_coords, triangles, attribute, uv_h, uv_w, c=3)
    return uv_position_map


def generate_dist(uv_g_t, uv_prnet, eye_dist):
    dist_mat = pixel_dist(uv_prnet, uv_g_t, eye_dist)
    return dist_mat



def generate_uv_g_t(transformed_vertices, colors, triangles, uv_coords):
    uv_h = uv_w = 256
    image_h = image_w = 256
    uv_coords = process_uv(uv_coords, uv_h, uv_w)

    #-- 1. uv texture map
    # attribute = colors
    # uv_texture_map = mesh.render.render_colors(uv_coords, triangles, attribute, uv_h, uv_w, c=3)
    # io.imsave('{}/uv_texture_map.jpg'.format(save_folder), np.squeeze(uv_texture_map))

    #-- 2. uv position map in 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network'
    #--   for face reconstruction & alginment(dense correspondences)
    # To some extent, when uv space is regular, position map is a subclass of geometry image(recording geometry information in regular image)
    # Notice: position map doesn't exit alone, it depends on the corresponding rendering(2d facical image).
    # Attribute is the position with respect to image coords system.
    # use standard camera & orth projection here
    projected_vertices = transformed_vertices.copy()
    image_vertices = mesh.transform.to_image(
        projected_vertices, image_h, image_w)
    position = image_vertices.copy()
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
    attribute = position
    # corresponding 2d facial image
    # image = mesh.render.render_colors(image_vertices, triangles, colors, image_h, image_w, c=3)
    uv_position_map = render.render_colors(
        uv_coords, triangles, attribute, uv_h, uv_w, c=3)
    return uv_position_map



def vertex2uvmap(vertex, triangles, uv_coords):
    uv_h = uv_w = 256
    image_h = image_w = 256
    uv_coords = uv_coords.copy()
    uv_coords = process_uv(uv_coords, uv_h, uv_w)

    #-- 1. uv texture map
    # attribute = colors
    # uv_texture_map = mesh.render.render_colors(uv_coords, triangles, attribute, uv_h, uv_w, c=3)
    # io.imsave('{}/uv_texture_map.jpg'.format(save_folder), np.squeeze(uv_texture_map))

    #-- 2. uv position map in 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network'
    #--   for face reconstruction & alginment(dense correspondences)
    # To some extent, when uv space is regular, position map is a subclass of geometry image(recording geometry information in regular image)
    # Notice: position map doesn't exit alone, it depends on the corresponding rendering(2d facical image).
    # Attribute is the position with respect to image coords system.
    # use standard camera & orth projection here

    # corresponding 2d facial image
    # image = mesh.render.render_colors(image_vertices, triangles, colors, image_h, image_w, c=3)
    uv_position_map = render.render_colors_prnet(
        uv_coords, triangles.T, vertex.T, uv_h, uv_w, c=3)
    return uv_position_map


def uvmap2vertex(uvmap, uv_coords):
    uv_h = uv_w = 256
    uv_coords = uv_coords.copy()
    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    uv_coords = np.round(uv_coords).astype(np.int16)
    vertex = uvmap[uv_coords[:,1], uv_coords[:,0], :]

    #x = np.arange(0, uv_w)
    #y = np.arange(0, uv_h)
    #xx, yy = np.meshgrid(x, y)
    #uvmap1 = uvmap[:,:,0]
    #f1 = interpolate.interp2d(xx,yy,uvmap1, kind = 'linear')
    #vertex = f1(uv_coords[0,:], uv_coords[1,:])

    return vertex


def load_uv_coords(path = 'BFM_UV.mat'):
    ''' load uv coords of BFM
    Args:
        path: path to data.
    Returns:
        uv_coords: [nver, 2]. range: 0-1
    '''
    C = sio.loadmat(path)
    uv_coords = C['UV'].copy(order = 'C')
    return uv_coords