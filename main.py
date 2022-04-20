import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mobilenet_v1
import numpy as np
import cv2
import dlib
from MM3D.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from MM3D.inference import get_suffix, parse_roi_box_from_landmark, crop_img,\
     predict_68pts,predict_dense, parse_roi_box_from_bbox
from MM3D.estimate_pose import parse_pose
import argparse
import torch.backends.cudnn as cudnn
from network import PRNet_PAF_Vis_Shape as net
from datasets import  ToTensor
from MM3D.uvmap import uvmap as uvmodule
from MM3D.params import *
from MM3D.inference import predict_3DMM_paras
import scipy.io as sio
from MM3D.utils import NormDirection
import imageio
from MM3D.paf import gen_img_paf_from_vertex

STD_SIZE = 120

gpu = 0
if gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'
    resume_path = 'models/_checkpoint_epoch_22.pth.tar'
    mean_shape = sio.loadmat('./MM3D/model/mean_shape.mat')['vertex']
    tri_full = sio.loadmat('./MM3D/model/tri_full.mat')['tri_full']
    uv_coords = uvmodule.load_uv_coords('./MM3D/uvmap/BFM_UV.mat')

    shape_min = sio.loadmat('meta/shape_distribute.mat')['shape_min']
    shape_max = sio.loadmat('meta/shape_distribute.mat')['shape_max']

    des_path = 'examples/results/'

    shapediff_scale = 1e-3
    numVertex = mean_shape.shape[1]

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()

    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    model1 = net(in_channels=3, out_channels=3)
    if gpu:
        model1 = nn.DataParallel(model1).cuda()

    if os.path.isfile(resume_path):
        if gpu:
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)['state_dict']
            model1.load_state_dict(checkpoint)
        else:# class one(object):
            model1_dict = model1.state_dict()
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)['state_dict']
            for k in checkpoint.keys():
                model1_dict[k.replace('module.', '')] = checkpoint[k]
                model1.load_state_dict(model1_dict)
    else:
        print("no checkpoint at {resume_path}")
        return
    model1.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    for img_fp in args.files:
        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)
            vertices = predict_dense(param, roi_box)
            vertices_lst.append(vertices)
            R,t3d,alpha_shp,alpha_exp = predict_3DMM_paras(param,roi_box,img_ori.shape)
            file , filename = img_fp.rsplit('/', 1)
            filename = filename[0:-4]
            img = imageio.imread(img_fp)
            shape_dt = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
            ddfa_offset = (w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
            vertex_dt = R @ shape_dt + np.tile(t3d, (1, numVertex))
            height = img.shape[0]
            vertex_dt[1, :] = height - vertex_dt[1, :]

            # ToTensor
            transform = transforms.Compose([ToTensor()])
            # PAF feature
            paf_feature = gen_img_paf_from_vertex(img, vertex_dt, tri_full, uv_coords, kernel_size=3)
            paf_feature = np.array(paf_feature)
            paf_feature = paf_feature.astype(np.float32) / 255
            paf_feature = transform(paf_feature)
            paf_feature = paf_feature.unsqueeze(0)
            # vis map
            norm_vertex = NormDirection(vertex_dt.T, tri_full.T).T
            norm_vertex = (norm_vertex + 1) / 2
            norm_vertex[0, :] = norm_vertex[2, :]
            norm_vertex[1, :] = norm_vertex[2, :]
            visible_uv = uvmodule.vertex2uvmap(norm_vertex, tri_full, uv_coords)  # shape: (256, 256, 3)
            visible_uv = visible_uv[:, :, 0]
            visible_uv = visible_uv[:, :, None]
            visible_uv = transform(visible_uv)
            visible_uv = visible_uv.unsqueeze(0)
            # shape map
            shape_map = uvmodule.vertex2uvmap(ddfa_offset, tri_full, uv_coords)
            shape_map = (shape_map - shape_min) / (shape_max - shape_min)
            shape_map[shape_map > 1] = 1
            shape_map[shape_map < 0] = 0
            shape_map = transform(shape_map)
            shape_map = shape_map.unsqueeze(0)

            outputs = model1(paf_feature, visible_uv, shape_map)
            outputs = outputs.cpu().detach().numpy()
            output = outputs[0, :, :, :]
            output = np.transpose(output, (1, 2, 0)) / shapediff_scale
            shapediff = uvmodule.uvmap2vertex(output, uv_coords).T
            shape_3ddfa = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
            shape_dt = shape_3ddfa + shapediff
            sio.savemat(des_path + filename + '.mat', {'shape_dt': shape_dt})
            ind += 1

    print('The result of ' + filename + ' are in examples/results')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='true', type=str2bool)
    parser.add_argument('--dump_depth', default='true', type=str2bool)
    parser.add_argument('--dump_pncc', default='true', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)