import numpy as np
import cv2
import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
import matplotlib
import logging
import argparse
import time
import json
import os
import os.path as osp
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from kornia.geometry.transform import warp_perspective

from lightglue import LightGlue, SuperPoint, ALIKED

np.set_printoptions(suppress=True)  # 去掉e表示


class DataIOWrapper(torch.nn.Module):
    ''' Pre-propcess data from different sources '''

    def __init__(self, model):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
        self.model = model
        self.model = self.model.eval().to(self.device)

    def preprocess_image(self, img, gray_scale=True):
        # xoftr takes grayscale input images
        if gray_scale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if len(img.shape) == 2:  # grayscale image
            img = torch.from_numpy(img)[None][None].to(self.device).float() / 255.0
        else:  # Color image
            img = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        return img

    def from_cv_imgs(self, img0, img1):
        img0_tensor = self.preprocess_image(img0)
        img1_tensor = self.preprocess_image(img1)
        mkpts0, mkpts1, mconf, match_time = self.match_images(img0_tensor, img1_tensor)

        matches = np.concatenate([mkpts0, mkpts1], axis=1)
        data = {'matches': matches,
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'mconf': mconf,
                'img0': img0,
                'img1': img1,
                'match_time': match_time
                }

        return data

    def from_paths(self, img0_pth, img1_pth, read_color=False):
        imread_flag = cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE
        img0 = cv2.imread(img0_pth, imread_flag)
        img1 = cv2.imread(img1_pth, imread_flag)
        return self.from_cv_imgs(img0, img1)

    def match_images(self, image_l, image_r):
        torch.cuda.synchronize()
        start = time.time()
        pred = self.model(image_l, image_r)
        torch.cuda.synchronize()
        match_1 = time.time()
        match_time = match_1 - start

        mkpts0 = pred['keypoints0'].cpu().numpy()
        mkpts1 = pred['keypoints1'].cpu().numpy()
        matching_scores0 = pred['matching_scores'].detach().cpu().numpy()

        mconf = matching_scores0

        return mkpts0, mkpts1, mconf, match_time


def load_model(method, args, use_path=True):
    if use_path:
        matcher = eval(f"load_{method}")(args)
        return matcher.from_paths
    else:
        matcher = eval(f"load_{method}")(args)
        return matcher.from_cv_imgs


def rbd(data: dict) -> dict:
    """ Remove batch dimension from elements in data """
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def load_sp_lg(args):

    class Matching(torch.nn.Module):
        def __init__(self, sp_conf, lg_conf):
            super().__init__()
            device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
            self.extractor = SuperPoint(**sp_conf).eval().to(device)  # load the feature extractor
            self.matcher = LightGlue(features='superpoint', **lg_conf).eval().to(device)  # load the matcher
            n_layers = lg_conf['n_layers']
            # print(f"n_layers: {n_layers}")
            ckpt_path = args.ckpt
            # rename old state dict entries
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            for i in range(n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.matcher.load_state_dict(state_dict, strict=False)

        def forward(self, image0, image1):

            feats0 = self.extractor.extract(image0, resize=None)  # auto-resize the image, disable with resize=None
            feats1 = self.extractor.extract(image1, resize=None)

            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
            matching_scores0 = matches01['matching_scores0']
            matching_scores = matching_scores0[matches[..., 0]]

            return {'matching_scores': matching_scores, 'keypoints0': points0, 'keypoints1': points1}
    # descriptor_dim = 256, nms_radius = 20, max_num_keypoints = 50, detection_threshold = 0.001, remove_borders = 4
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 30,
        "max_num_keypoints": 30,
        "detection_threshold": 0.003,
        "remove_borders": 4,
    }
    lg_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.99,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
    matcher = Matching(sp_conf, lg_conf)
    matcher = DataIOWrapper(matcher)
    return matcher


def load_al_lg(args):

    class Matching(torch.nn.Module):
        def __init__(self, al_conf, lg_conf):
            super().__init__()
            device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
            self.extractor = ALIKED(**al_conf).eval().to(device)  # load the feature extractor
            self.matcher = LightGlue(features='aliked', **lg_conf).eval().to(device)  # load the matcher
            n_layers = lg_conf['n_layers']
            # print(f"n_layers: {n_layers}")
            ckpt_path = 'weights/aliked-n16.pth'
            # rename old state dict entries
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
            for i in range(n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.matcher.load_state_dict(state_dict, strict=False)

        def forward(self, image0, image1):

            feats0 = self.extractor.extract(image0, resize=None)  # auto-resize the image, disable with resize=None
            feats1 = self.extractor.extract(image1, resize=None)

            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
            matching_scores0 = matches01['matching_scores0']
            matching_scores = matching_scores0[matches[..., 0]]

            return {'matching_scores': matching_scores, 'keypoints0': points0, 'keypoints1': points1}

    al_conf = {
        "model_name": "aliked-n16",
        "max_num_keypoints": 30,  # 2048
        "detection_threshold": 0.2,  # 0.0002
        "nms_radius": 20,
    }
    lg_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.001,  # 0.1  # match threshold
        "weights": None,
    }
    matcher = Matching(al_conf, lg_conf)
    matcher = DataIOWrapper(matcher)
    return matcher


def load_model_optional(args):
    """ 加载模型 NOTE: 算法比较 """
    from exp_model_config import sp_conf, lg_conf_sp, al_conf, lg_conf_al

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    if args.method == 'sp_lg':
        extractor = SuperPoint(**sp_conf).eval().to(device)
        matcher = LightGlue(features='superpoint', **lg_conf_sp).eval().to(device)
        n_layers = lg_conf_sp['n_layers']

    elif args.method == 'al_lg':
        extractor = ALIKED(**al_conf).eval().to(device)
        matcher = LightGlue(features='aliked', **lg_conf_al).eval().to(device)
        n_layers = lg_conf_al['n_layers']

    elif args.method == 'allg':
        # extractor = ALIKED(model_name='aliked-n16', detection_threshold=0.2, nms_radius=20).eval().to(device)
        extractor = ALIKED(model_name='aliked-n16', max_num_keypoints=20, detection_threshold=0.15, nms_radius=20).eval().to(device)
        matcher = LightGlue(features="aliked", filter_threshold=0.001).eval().to(device)
        n_layers = lg_conf_al['n_layers']
    else:
        raise ValueError(f'Unknown model: {args.method}')

    ckpt_path = args.ckpt
    # rename old state dict entries
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    for i in range(n_layers):
        pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    matcher.load_state_dict(state_dict, strict=False)

    return extractor, matcher


def initial_matching(args, path_l=None, path_r=None):
    """
    初始匹配 返回加载模型匹配的初始结果
    return: img0_color, img1_color, mkpts0, mkpts1, mconf, color
    """
    if path_l is not None and path_r is not None:
        args.fig0 = path_l
        args.fig1 = path_r
    if args.fig0 is None or args.fig1 is None:
        raise ValueError("图像未找到!")
    print(f'args.fig0: {args.fig0}\nargs.fig1: {args.fig1}')

    img0_color = cv2.imread(args.fig0)
    img1_color = cv2.imread(args.fig1)
    img0_color = cv2.cvtColor(img0_color, cv2.COLOR_BGR2RGB)
    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)

    matcher = load_model(args.method, args)
    match_res = matcher(args.fig0, args.fig1)
    # dict_keys(['matches', 'mkpts0', 'mkpts1', 'mconf', 'img0', 'img1', 'match_time'])
    # print(f"match_res.keys: {match_res.keys()}")

    mkpts0 = match_res['mkpts0']
    mkpts1 = match_res['mkpts1']
    mconf = match_res['mconf']
    # print('mconf:\n', mconf)
    if len(mconf) > 0:
        conf_min = mconf.min()
        conf_max = mconf.max()
        mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)
    color = matplotlib.cm.jet(mconf)

    # 按照y坐标进行排序 NOTE: 排序 为什么 匹配结果 有问题
    # mkpts0 = np.array(sorted(mkpts0, key=lambda x: x[1]))
    # mkpts1 = np.array(sorted(mkpts1, key=lambda x: x[1]))

    return img0_color, img1_color, mkpts0, mkpts1, mconf, color


def camera_considered(mkpts0, mkpts1):
    """ 加载双目相机参数 校正匹配点畸变 计算三维坐标 """
    from calib_para.stereo_config import stereo_camera

    K1 = stereo_camera().K_left
    K2 = stereo_camera().K_right
    dist1 = stereo_camera().dist_left
    dist2 = stereo_camera().dist_right
    R = stereo_camera().R
    T = stereo_camera().T.reshape(3, 1)

    # 内存连续化 提高速度
    mkpts0 = np.ascontiguousarray(mkpts0)
    mkpts1 = np.ascontiguousarray(mkpts1)
    # mkpts0 = np.array(mkpts0)
    # mkpts1 = np.array(mkpts1)

    # 校正特征点坐标
    mkpts0_rectified = cv2.undistortPoints(mkpts0, K1, dist1, P=K1)
    mkpts1_rectified = cv2.undistortPoints(mkpts1, K2, dist2, P=K2)

    # 投影矩阵 P1 = K1[I∣0], P2 = K2[R∣t]
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(-1, 1)))
    # print(f'P1:\n{P1}\nP2:\n{P2}')

    # 三角测量
    points3D_homogeneous = cv2.triangulatePoints(P1, P2, mkpts0_rectified, mkpts1_rectified)
    points_3d = cv2.convertPointsFromHomogeneous(points3D_homogeneous.T)
    # print(f'3D Points:\n{points_3d.squeeze()}')
    points_3d = points_3d.squeeze()
    assert points_3d.shape == (len(mkpts0), 3), "points_3d 必须是 (N, 3) 的形状"

    # 返回字典
    cam_considered_dict = {
        'K1': K1,
        'K2': K2,
        'dist1': dist1,
        'dist2': dist2,
        'R': R,
        'T': T,
        'P1': P1,
        'P2': P2,
        'is_rectified': True,
        'mkpts0_rectified': mkpts0_rectified,
        'mkpts1_rectified': mkpts1_rectified,
        'points_3d': points_3d,
    }

    return cam_considered_dict


def args_parse_mqq(description: str):
    """ 命令行解析 加载参数 """

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--method', type=str, choices=["xoftr", 'sp_lg', 'al_lg', 'allg'], help="")

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    if description == 'demo matching':
        parser.add_argument('--fig0', type=str, default='', help='')
        parser.add_argument('--fig1', type=str, default='', help='')
    elif description == 'batch experiment':
        parser.add_argument('--refiner', type=str2bool, default='False', help='Use refiner or not.')
        parser.add_argument('--subpixel', type=str2bool, default='False', help='Use subpixel or not.')
        parser.add_argument('--input_l', type=str, default='', help='Image L directory.')
        parser.add_argument('--input_r', type=str, default='', help='Image R directory.')
    else:
        raise ValueError(f"Unknown description: {description}")

    # parse_known_args()函数返回的是一个元组, 其中包含了传递给命令的参数信息和剩余的参数信息
    args, remaining_args = parser.parse_known_args()

    if args.method == 'xoftr':
        parser.add_argument('--match_threshold', type=float, default=0.3)
        parser.add_argument('--fine_threshold', type=float, default=0.1)
        parser.add_argument('--ckpt', type=str, default="./weights/minima_xoftr.ckpt")
    elif args.method == 'sp_lg':
        parser.add_argument('--ckpt', type=str, default="./weights/minima_lightglue.pth")
    elif args.method == 'al_lg':
        parser.add_argument('--ckpt', type=str, default="./weights/minima_lightglue.pth")  # ./weights/minima_lightglue.pth  ./weights/aliked-n16.pth
    elif args.method == 'allg':
        parser.add_argument('--ckpt', type=str, default="./weights/aliked-n16.pth")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    args = parser.parse_args()
    print("")
    # parse_args()函数返回的是一个命名空间(NameSpace)对象, 其中包含了传递给命令的参数信息
    logging.info(f'args: {args}')

    return args, remaining_args


if __name__ == '__main__':

    args, _ = args_parse_mqq(description='demo matching')  # args

    # img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args, 'input/L2.bmp', 'input/R2.bmp')
    img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args)

    print(f'mkpts0: {mkpts0.shape}\n{mkpts0}\n\nmkpts1: {mkpts1.shape}\n{mkpts1}')

    print("****************** Matching Model Done! ******************")
