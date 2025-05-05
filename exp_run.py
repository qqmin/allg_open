import argparse
import cv2
import glob
import json
import logging
import matplotlib
import numpy as np
import os
import os.path as osp
import time
import torch
import warnings
from collections import defaultdict, OrderedDict
from kornia.geometry.transform import warp_perspective
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from exp_model import rbd, load_model_optional, args_parse_mqq, initial_matching, camera_considered
from exp_refiner import MatcherRefiner, VisualizerRefiner
from exp_subpixel import Matching_Subpixel_Refiner, Subpixel_Refiner_View, Subpixel_Refiner_Eval
from exp_viewer import opencv_view_original, opencv_view_after, opencv_view_compare, epipolar_error_view
from lightglue import SuperPoint, LightGlue, ALIKED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageLoader(object):
    def __init__(self, filepath: str):
        self.images = glob.glob(os.path.join(filepath, '*.png')) + \
            glob.glob(os.path.join(filepath, '*.jpg')) + \
            glob.glob(os.path.join(filepath, '*.bmp'))
        self.images.sort()  # 按文件名排序
        self.N = len(self.images)
        logging.info(f'Loading {self.N} images')
        self.mode = 'images'

    def __getitem__(self, item):
        filename = self.images[item]  # 获取第 item 个图片的路径
        img = cv2.imread(filename)
        if img is None:
            logging.warning(f"Failed to read image: {filename}")
            return None, None  # 或者抛出异常
        print('filename:', filename)

        return img, filename

    def __len__(self):
        return self.N


def _refiner(img0, img1, mkpts0, mkpts1):
    ''' 匹配点优化 '''
    refiner = MatcherRefiner()
    mkpts0_refined, mkpts1_refined = refiner.refine_matches(
        mkpts0, mkpts1, y_thresh=50, ransac_thresh=30, ransac_confidence=0.99,
        max_iters=5000, epipolar_thresh=20, depth_z_thresh=80)

    # print("****************** exp refiner done! ******************")

    return mkpts0_refined, mkpts1_refined


def _subpixel(img0, img1, mkpts0, mkpts1):
    mkpts0_ = mkpts0.copy()
    mkpts1_ = mkpts1.copy()
    refiner = Matching_Subpixel_Refiner(img0, img1, mkpts0_, mkpts1_)
    mkpts0_subpixel, mkpts1_subpixel = refiner.refine_subpixel_cv((5, 5), (-1, -1), 50, 0.001)

    _mkpts0_ = mkpts0.copy()
    _mkpts1_ = mkpts1.copy()

    ''' 亚像素细化评估'''
    evaluator = Subpixel_Refiner_Eval(_mkpts0_, _mkpts1_, mkpts0_subpixel, mkpts1_subpixel)
    # evaluator.plot_displacement_histogram()
    try:
        metrics = evaluator.compute_metrics()
        print("==== 亚像素优化效果评估 ====")
        for k, v in metrics.items():
            print(f"{k:20}: {v:.4f}")

        # evaluator.plot_displacement_vectors(img0, img1, scale_factor=20)

    except ValueError as e:
        print(f"评估失败: {str(e)}")

    # print("****************** exp subpixel done! ******************")

    return mkpts0_subpixel, mkpts1_subpixel


def algorithm_compare(image_l, image_r):
    from exp_model import rbd
    # 加载ALIKED和LightGlue
    # extractor = ALIKED().eval().to(device)  # load the extractor
    extractor = ALIKED(model_name='aliked-n16', detection_threshold=0.2, nms_radius=20).eval().to(device)
    # extractor = SuperPoint().eval().to(device) # TODO 报错 AssertionError
    # matcher = LightGlue(features="aliked", filter_threshold=0.03).eval().to(device)
    # matcher = LightGlue(features="aliked", filter_threshold=0.005).eval().to(device)
    matcher = LightGlue(features="aliked", filter_threshold=0.001).eval().to(device)

    image_l = cv2.imread(image_l)
    image_r = cv2.imread(image_r)
    # numpy to tensor
    # image_l_T = torch.from_numpy(image_l).permute(2, 0, 1).float()
    # image_r_T = torch.from_numpy(image_r).permute(2, 0, 1).float()
    # numpy_image_to_torch  NOTE: 对齐 allg
    from copy import deepcopy
    image_l_ = deepcopy(image_l).transpose((2, 0, 1))  # HxWxC to CxHxW
    image_r_ = deepcopy(image_r).transpose((2, 0, 1))
    image_l_T = torch.tensor(image_l_ / 255.0, dtype=torch.float)
    image_r_T = torch.tensor(image_r_ / 255.0, dtype=torch.float)

    feats0 = extractor.extract(image_l_T.to(device))
    feats1 = extractor.extract(image_r_T.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    matching_scores0 = matches01['matching_scores0']
    matching_scores = matching_scores0[matches[..., 0]]

    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    matches = matches.cpu()
    m_kpts0 = m_kpts0.cpu().numpy()
    m_kpts1 = m_kpts1.cpu().numpy()

    return image_l, image_r, kpts0, kpts1, m_kpts0, m_kpts1


def main_demo():

    parser = argparse.ArgumentParser(description='main_demo')
    parser.add_argument('--method', type=str, choices=["xoftr", 'sp_lg', 'allg'], help="")

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    parser.add_argument('--refiner', type=str2bool, default='False', help='Use refiner or not.')
    parser.add_argument('--subpixel', type=str2bool, default='False', help='Use subpixel or not.')
    parser.add_argument('--fig0', type=str, default='', help='')
    parser.add_argument('--fig1', type=str, default='', help='')

    args, remaining_args = parser.parse_known_args()

    if args.method == 'xoftr':
        parser.add_argument('--match_threshold', type=float, default=0.3)
        parser.add_argument('--fine_threshold', type=float, default=0.1)
        parser.add_argument('--ckpt', type=str, default="./weights/minima_xoftr.ckpt")
    elif args.method == 'sp_lg' or args.method == 'allg':
        parser.add_argument('--ckpt', type=str, default="./weights/minima_lightglue.pth")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    args = parser.parse_args()
    print("")
    logging.info(f'args: {args}')

    img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args)
    print(f'mkpts0: {mkpts0.shape}\nmkpts1: {mkpts1.shape}')

    mkpts0_ = mkpts0.copy()   # NOTE: !!! 一定要加 不然不显示
    mkpts1_ = mkpts1.copy()

    tt = time.time()

    # 查看原始匹配
    opencv_view_original(img0_color, img1_color, mkpts0, mkpts1, color, args)
    # 匹配优化
    mkpts0_refined, mkpts1_refined = _refiner(img0_color, img1_color, mkpts0_, mkpts1_)

    # 按y坐标排序
    mkpts0_refined = mkpts0_refined[mkpts0_refined[:, 1].argsort()]
    mkpts1_refined = mkpts1_refined[mkpts1_refined[:, 1].argsort()]
    mkpts0_refined_ = mkpts0_refined.copy()
    mkpts1_refined_ = mkpts1_refined.copy()

    # 亚像素优化
    mkpts0_subpixel, mkpts1_subpixel = _subpixel(img0_color, img1_color, mkpts0_refined_, mkpts1_refined_)

    mkpts0_subpixel_ = mkpts0_subpixel.copy()
    mkpts1_subpixel_ = mkpts1_subpixel.copy()

    # 查看优化后
    opencv_view_after(img0_color, img1_color, mkpts0_subpixel_, mkpts1_subpixel_, args)

    # 计算 epipolar error
    epipolar_error_view(img0_color, img1_color, mkpts0_subpixel_, mkpts1_subpixel_, output='output/main_demo_epipolar_errors.png')

    print(f"Elapsed time: {time.time() - tt}")


def main_batch():
    args, _ = args_parse_mqq(description='batch experiment')  # args

    image_loader_l = ImageLoader(args.input_l)
    image_loader_r = ImageLoader(args.input_r)

    # 建立数据保存文件夹 如果目录不存在就创建 如果已经存在就不做任何事情
    dir_paths = ['output_tmp/mkpts0', 'output_tmp/mkpts1', 'output_tmp/match_img', 'output_tmp/epipolar_errors']
    for path in dir_paths:
        os.makedirs(path, exist_ok=True)

    logging.info("----------> 按 'Space' 开始. \t按 'q' 或 'ESC' 停止. <----------")

    for i in range(0, len(image_loader_l)):
        print(f'\n******************************************* 循环({(i):03d})开始 *******************************************')
        t1 = time.time()  # 记录开始时间
        img_l, path_l = image_loader_l[i]
        img_r, path_r = image_loader_r[i]
        assert img_l is not None and img_r is not None, 'img_l or img_r is None'
        imgl_bgr = cv2.imread(path_l)
        imgr_bgr = cv2.imread(path_r)
        imgl_rgb = cv2.cvtColor(imgl_bgr, cv2.COLOR_BGR2RGB)
        imgr_rgb = cv2.cvtColor(imgr_bgr, cv2.COLOR_BGR2RGB)
        print(f'img_l:{img_l.shape}: {path_l}')
        print(f'img_r:{img_r.shape}: {path_r}')

        # 粗匹配 NOTE: 算法比较
        # img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args, path_l, path_r)
        img0_color, img1_color, kpts0, kpts1, mkpts0, mkpts1 = algorithm_compare(path_l, path_r)
        print(f'mkpts0: {mkpts0.shape}, mkpts1: {mkpts1.shape}')
        t2 = time.time()  # 记录粗匹配时间 t = t2 - t1
        # 匹配优化
        mkpts0_ = mkpts0.copy()
        mkpts1_ = mkpts1.copy()
        if args.refiner:
            logging.info(f"-----> args.refiner: {args.refiner}, 执行匹配优化... <-----")
            mkpts0_refined, mkpts1_refined = _refiner(img0_color, img1_color, mkpts0_, mkpts1_)
            print(f"---> 循环({(i):03d})优化完成: 优化/原始匹配数: {len(mkpts0_refined)}/{len(mkpts0_)} <---")
        else:
            logging.info(f"-----> args.refiner: {args.refiner}, 不执行匹配优化... <-----")
            # mkpts0_refined, mkpts1_refined = mkpts0_, mkpts1_
            """ 坐标控制过滤 """
            final_matches = []
            # for y1, y2 in zip(mkpts0_[:, 1], mkpts1_[:, 1]):
            for (x1, y1), (x2, y2) in zip(mkpts0_, mkpts1_):
                # if y1 < 1930 and y2 < 1930 and not (1885 < y1 < 1895):
                if y1 < 1945 and y2 < 1945 and (y2 <= 1500 or x2 <= 890):
                    final_matches.append(True)
                else:
                    final_matches.append(False)
            mkpts0_ = mkpts0_[final_matches]
            mkpts1_ = mkpts1_[final_matches]
            mask = np.abs(mkpts0_[:, 1] - mkpts1_[:, 1]) <= 50
            mkpts0_refined, mkpts1_refined = mkpts0_[mask], mkpts1_[mask]

        # 按y坐标排序
        mkpts0_refined = mkpts0_refined[mkpts0_refined[:, 1].argsort()]
        mkpts1_refined = mkpts1_refined[mkpts1_refined[:, 1].argsort()]
        mkpts0_refined_ = mkpts0_refined.copy()
        mkpts1_refined_ = mkpts1_refined.copy()
        t3 = time.time()  # 记录匹配优化时间 t = t3 - t2
        # 亚像素优化
        if args.subpixel:
            logging.info(f"-----> args.subpixel: {args.subpixel}, 执行亚像素优化... <-----")
            mkpts0_subpixel, mkpts1_subpixel = _subpixel(img0_color, img1_color, mkpts0_refined_, mkpts1_refined_)
            print(f"---> 循环({(i):03d})亚像素优化完成 <---")
        else:
            logging.info(f"-----> args.subpixel: {args.subpixel}, 不执行亚像素优化... <-----")
            mkpts0_subpixel, mkpts1_subpixel = mkpts0_refined_, mkpts1_refined_

        mkpts0_subpixel_ = mkpts0_subpixel.copy()
        mkpts1_subpixel_ = mkpts1_subpixel.copy()
        t4 = time.time()  # 记录亚像素优化时间 t = t4 - t3
        status = f"img_num: {i+1}/{len(image_loader_l)}  mkpts0_subpixel_/mkpts0: {len(mkpts0_subpixel_)}/{len(mkpts0)} mkpts1_subpixel_/mkpts1: {len(mkpts1_subpixel_)}/{len(mkpts1)}"

        # 查看优化前
        # logging.info(f"-----> 执行查看优化前图片... <-----")
        # out, mkpts0_xy, mkpts1_xy = opencv_view_original(img0_color, img1_color, mkpts0, mkpts1, color, args, demo=False)
        # 查看优化后
        logging.info(f"-----> 执行查看优化后图片... <-----")
        out, mkpts0_xy, mkpts1_xy = opencv_view_after(img0_color, img1_color, mkpts0_subpixel_, mkpts1_subpixel_, args, demo=False)

        # 计算 epipolar error
        logging.info(f"-----> 计算 epipolar error... <-----")
        # epipolar_error_view(img0_color, img1_color, mkpts0_subpixel_, mkpts1_subpixel_, output=f'{dir_paths[3]}/epipolar_errors_{i:03d}.png', demo=False)
        # epipolar_error_view(img0_color, img1_color, mkpts0_subpixel_, mkpts1_subpixel_, demo=False)

        print('-'*90 + '\n' + ' '*8 + status + '\n' + '-'*90)
        np.savetxt(f'{dir_paths[0]}/xy_{i:03d}.txt', mkpts0_xy, fmt='%.4f')
        np.savetxt(f'{dir_paths[1]}/xy_{i:03d}.txt', mkpts1_xy, fmt='%.4f')
        if len(mkpts0_xy) == len(mkpts0_subpixel_) or len(mkpts1_xy) == len(mkpts1_subpixel_):
            print('len(mkpts0_xy) == len(refined_left) or len(mkpts1_xy) == len(refined_right)')
            cv2.imwrite(f'{dir_paths[2]}/detect_match_{i:03d}.jpg', cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            print(f'save image: {dir_paths[2]}/detect_match_{i:03d}.jpg')
        else:
            print('len(mkpts0_xy) =?= len(refined_left) or len(mkpts1_xy) =?= len(refined_right)')
        logging.info(f"---> Epoch {(i):03d} time: {time.time()-t1:.4f} s (model:{t2-t1:.4f} s, refiner:{t3-t2:.4f} s, subpixel:{t4-t3:.4f} s) <---")
        print(f'******************************************* 循环({(i):03d})结束 *******************************************')
        cv2.namedWindow('batch detection matches', 0)
        cv2.resizeWindow('batch detection matches', 1500, 1110)
        cv2.moveWindow('batch detection matches', 1000, 0)
        cv2.setWindowTitle('batch detection matches', f'detection matches: {status}')
        cv2.imshow('batch detection matches', out)
        c = cv2.waitKey()
        if c == ord('q') or c == 27:
            break


if __name__ == '__main__':
    # main_demo()
    main_batch()
