import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from scipy import stats
import argparse
import logging


class Matching_Subpixel_Refiner:
    def __init__(self, img_l, img_r, mkpts0, mkpts1):
        """ img_l,img_r: (H,W,3)/(H,W)   mkpts0,mkpts1: (N,2) """
        self.img_l = self._preprocess_image(img_l)
        self.img_r = self._preprocess_image(img_r)

        self.mkpts0 = mkpts0
        self.mkpts1 = mkpts1
        self.mkpts0_subpixel = None
        self.mkpts1_subpixel = None

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
        # img = cv2.equalizeHist(img)  # 直方图均衡化
        # img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    def refine_subpixel_cv(self, win_size=(5, 5), zero_zone=(-1, -1), max_iter=50, epsilon=0.001):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

        self.mkpts0_subpixel = cv2.cornerSubPix(self.img_l, self.mkpts0, winSize=win_size, zeroZone=zero_zone, criteria=criteria)
        self.mkpts1_subpixel = cv2.cornerSubPix(self.img_r, self.mkpts1, winSize=win_size, zeroZone=zero_zone, criteria=criteria)

        return self.mkpts0_subpixel, self.mkpts1_subpixel

    def refine_subpixel_flow(self, win_size=(21, 21), max_level=3):
        self.mkpts0_subpixel, status, _ = cv2.calcOpticalFlowPyrLK(self.img_l, self.img_r, self.mkpts0, None, win_size, max_level)
        self.mkpts1_subpixel, status, _ = cv2.calcOpticalFlowPyrLK(self.img_l, self.img_r, self.mkpts1, None, win_size, max_level)
        valid = (status.ravel() == 1)  # 双向一致性检查
        self.mkpts0_subpixel = self.mkpts0_subpixel[valid]
        self.mkpts1_subpixel = self.mkpts1_subpixel[valid]

        return self.mkpts0_subpixel, self.mkpts1_subpixel


class Subpixel_Refiner_View():

    def __init__(self, img_l, img_r, mkpts0, mkpts1, mkpts0_subpixel, mkpts1_subpixel):
        self.img_l = img_l
        self.img_r = img_r
        self.mkpts0 = mkpts0
        self.mkpts1 = mkpts1
        self.mkpts0_subpixel = mkpts0_subpixel
        self.mkpts1_subpixel = mkpts1_subpixel

    def compare_view_matplotlib(self, outpath=None):

        img_l_rgb = self.img_l
        img_r_rgb = self.img_r

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(np.concatenate([img_l_rgb, img_r_rgb], axis=1))

        ax.scatter(self.mkpts0[:, 0], self.mkpts0[:, 1], marker='o', c='yellow', s=2, label='original left')
        ax.scatter(self.mkpts1[:, 0]+img_l_rgb.shape[1], self.mkpts1[:, 1], marker='o', c='yellow', s=2, label='original right')
        # 绘制亚像素点 使用精确浮点坐标
        ax.scatter(self.mkpts0_subpixel[:, 0], self.mkpts0_subpixel[:, 1], marker='o', c='red', s=2, label='subpixel left')
        ax.scatter(self.mkpts1_subpixel[:, 0]+img_l_rgb.shape[1], self.mkpts1_subpixel[:, 1], marker='o', c='red', s=2, label='subpixel right')
        # 绘制连线
        for p_orig, p_sub in zip(self.mkpts0_subpixel, self.mkpts1_subpixel):
            plt.plot([p_orig[0], p_sub[0]+img_l_rgb.shape[1]], [p_orig[1], p_sub[1]], color='green', linewidth=1)
        plt.title('original vs subpixel')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(outpath, dpi=600)
        plt.show()

    def compare_view_opencv(self, line_color=(0, 255, 0), point_color=(255, 0, 0), point_size=300, line_thickness=100):
        """ NOTE opencv 绘制不了亚像素点 """
        img_l_rgb = self.img_l
        img_r_rgb = self.img_r
        vis_img = np.concatenate([img_l_rgb, img_r_rgb], axis=1)
        offset = img_l_rgb.shape[1]
        # 绘制亚像素点 NOTE opencv 绘制不了亚像素点
        for pt in self.mkpts0_subpixel:
            cv2.drawMarker(vis_img, tuple(np.round(pt).astype(int)), point_color, markerType=cv2.MARKER_CROSS, markerSize=point_size*2)
        for pt in self.mkpts1_subpixel:
            x, y = np.round(pt).astype(int)
            cv2.drawMarker(vis_img, (x+offset, y), point_color, markerType=cv2.MARKER_CROSS, markerSize=point_size*2)
        # 绘制连线
        for pt0, pt1 in zip(self.mkpts0_subpixel, self.mkpts1_subpixel):
            x0, y0 = np.round(pt0).astype(int)
            x1, y1 = np.round(pt1).astype(int)
            cv2.line(vis_img, (x0, y0), (x1+offset, y1), line_color, line_thickness)
        cv2.imshow('Stereo Correspondence', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compare_view_opencv_pro(self, image, corners_before, corners_after):
        num_points = corners_before.shape[0]
        for i in range(num_points):
            pt_before = corners_before[i]
            pt_after = corners_after[i]
            cv2.circle(image, (int(round(pt_before[0])), int(round(pt_before[1]))), 3, (0, 255, 255), 1)
            cv2.circle(image, (int(round(pt_after[0])), int(round(pt_after[1]))), 3, (0, 0, 255), -1)

            start_pt = (int(round(pt_before[0])), int(round(pt_before[1])))
            end_pt = (int(round(pt_after[0])), int(round(pt_after[1])))
            # cv2.arrowedLine(image, start_pt, end_pt, (0, 255, 0), 1, tipLength=0.3)
        return image  # 绘制了角点及箭头的图像 (BGR格式)

    def _zoom_area_opencv(self, image, points, win_size=15, zoom=10):

        h, w = image.shape[:2]
        display_img = image.copy()

        for pt in points:
            x, y = pt
            x_min = max(0, int(x)-win_size)
            x_max = min(w, int(x)+win_size+1)
            y_min = max(0, int(y)-win_size)
            y_max = min(h, int(y)+win_size+1)
            # 创建局部ROI
            roi = image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue
            # 放大局部区域
            big_roi = cv2.resize(roi, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            # 在放大图像中绘制亚像素点
            dx = (x - x_min) * zoom
            dy = (y - y_min) * zoom
            cv2.drawMarker(big_roi, (int(dx), int(dy)), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=1, line_type=cv2.LINE_AA)
            # 将ROI插值回原尺寸显示
            small_roi = cv2.resize(big_roi, (x_max-x_min, y_max-y_min), interpolation=cv2.INTER_AREA)
            display_img[y_min:y_max, x_min:x_max] = small_roi

        return display_img

    def _zoom_area_matplotlib(self, image, center, win_size, zoom, output=None):

        plt.figure(figsize=(10, 10))
        x, y = center.astype(int)
        h, w = image.shape[:2]
        # 提取局部区域
        x0 = max(0, x - win_size)
        x1 = min(w, x + win_size)
        y0 = max(0, y - win_size)
        y1 = min(h, y + win_size)
        patch = image[y0:y1, x0:x1]
        plt.imshow(cv2.resize(patch, None, zoom, zoom, cv2.INTER_CUBIC), cmap='gray')
        plt.suptitle('zoom')
        plt.axis('off')
        if output is not None:
            plt.savefig(output, dpi=300)


class Subpixel_Refiner_Eval():

    def __init__(self, mkpts0, mkpts1, mkpts0_subpixel, mkpts1_subpixel):
        self.mkpts0 = mkpts0
        self.mkpts1 = mkpts1
        self.mkpts0_subpixel = mkpts0_subpixel
        self.mkpts1_subpixel = mkpts1_subpixel
        self.metrics = {}
        eps = 1e-6  # 添加最小扰动防止除零
        # 计算位移量 计算原始点和亚像素点之间的位移 欧式距离
        self.delta_left = np.linalg.norm(mkpts0_subpixel - mkpts0 + eps, axis=1)
        self.delta_right = np.linalg.norm(mkpts1_subpixel - mkpts1 + eps, axis=1)
        # 过滤微小位移(<0.01 像素视为噪声)
        valid = (self.delta_left > 0.001) & (self.delta_right > 0.001)
        self.delta_left = self.delta_left[valid]
        self.delta_right = self.delta_right[valid]

    def compute_metrics(self) -> Dict[str, float]:

        # 在计算前检查
        print(f"len(delta_left): {len(self.delta_left)}, len(delta_right): {len(self.delta_right)}")

        if len(self.delta_left) < 3 or len(self.delta_right) < 3:
            raise ValueError("有效位移点不足，无法计算有意义指标")

        metrics = {
            'mean_displacement': (np.mean(self.delta_left) + np.mean(self.delta_right))/2,
            'max_displacement': max(np.max(self.delta_left), np.max(self.delta_right)),
            'displacement_std': (np.std(self.delta_left) + np.std(self.delta_right))/2,
            'median_displacement': (np.median(self.delta_left) + np.median(self.delta_right))/2,

            'mean_delta_left': np.mean(self.delta_left),
            'median_delta_left': np.median(self.delta_left),
            'std_delta_left': np.std(self.delta_left),
            'max_delta_left': np.max(self.delta_left),

            'mean_delta_right': np.mean(self.delta_right),
            'median_delta_right': np.median(self.delta_right),
            'std_delta_right': np.std(self.delta_right),
            'max_delta_right': np.max(self.delta_right),
        }
        # 仅当有有效数据时计算相关性
        if len(self.delta_left) > 2:
            metrics['pearson_corr'] = stats.pearsonr(self.delta_left, self.delta_right)[0]
            metrics['spearman_corr'] = stats.spearmanr(self.delta_left, self.delta_right)[0]
        else:
            metrics.update({'pearson_corr': np.nan, 'spearman_corr': np.nan})
        # 添加统计检验
        if len(self.delta_left) > 30:
            t_test = stats.ttest_rel(self.delta_left, self.delta_right)
            metrics['t-test_pvalue'] = t_test.pvalue

        self.metrics = metrics
        return metrics

    def plot_displacement_histogram(self, output=None):
        """ 绘制原始点与亚像素点之间位移的直方图 """
        plt.figure(figsize=(10, 6))
        # bins (int): 直方图的柱数
        plt.hist(self.delta_left, bins=50, color='c', edgecolor='k', alpha=0.7, label='Left Image')
        plt.hist(self.delta_right, bins=50, color='c', edgecolor='k', alpha=0.7, label='Right Image')
        plt.xlabel('Displacement (pixels)')
        plt.ylabel('Frequency')
        plt.title('Subpixel Refinement Displacement Distribution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        # plt.show()

    def plot_displacement_vectors(self, img_l, img_r, scale_factor=20.0, output=None):
        """ 位移矢量可视化 """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        # 左图矢量场
        ax1.imshow(img_l)
        ax1.quiver(self.mkpts0[:, 0], self.mkpts0[:, 1],
                   (self.mkpts0_subpixel[:, 0]-self.mkpts0[:, 0])*scale_factor,
                   (self.mkpts0_subpixel[:, 1]-self.mkpts0[:, 1])*scale_factor,
                   angles='xy', scale_units='xy', scale=1, color='cyan', width=0.002)
        ax1.set_title(f"Displacement Vectors Left(*{scale_factor})")
        ax1.axis('off')
        # 右图矢量场
        ax2.imshow(img_r)
        ax2.quiver(self.mkpts1[:, 0], self.mkpts1[:, 1],
                   (self.mkpts1_subpixel[:, 0]-self.mkpts1[:, 0])*scale_factor,
                   (self.mkpts1_subpixel[:, 1]-self.mkpts1[:, 1])*scale_factor,
                   angles='xy', scale_units='xy', scale=1, color='cyan', width=0.002)
        ax2.set_title(f"Displacement Vectors Right(*{scale_factor})")
        ax2.axis('off')

        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        # plt.show()


if __name__ == '__main__':

    from exp_model import args_parse_mqq, initial_matching
    args, _ = args_parse_mqq(description='demo matching')  # args

    # img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args, 'input/L2.bmp', 'input/R2.bmp')
    img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args)
    print(f'mkpts0: {mkpts0.shape}\nmkpts1: {mkpts1.shape}')

    mkpts0_ = mkpts0.copy()   # NOTE: !!! 一定要加 不然不显示
    mkpts1_ = mkpts1.copy()

    ''' 匹配点优化 '''
    from exp_refiner import MatcherRefiner, VisualizerRefiner
    refiner = MatcherRefiner()
    mkpts0_refined, mkpts1_refined = refiner.refine_matches(mkpts0_, mkpts1_, y_thresh=50, ransac_thresh=7,
                                                            ransac_confidence=0.995, max_iters=2000,
                                                            epipolar_thresh=20, depth_z_thresh=50)

    print(f"优化匹配数/原始匹配数: {len(mkpts0_refined)}/{len(mkpts0)}")
    print(f"平均重投影误差: {refiner.compute_reprojection_error(mkpts0_refined, mkpts1_refined):.2f} 像素")

    # visualizer = VisualizerRefiner(img0_color, img1_color)
    # visualizer.visualize_comparison(method='opencv', original_matches=(mkpts0, mkpts1),
    #                                 refined_matches=(mkpts0_refined, mkpts1_refined))
    # visualizer.visualize_comparison(method='matplotlib', original_matches=(mkpts0, mkpts1),
    #                                 refined_matches=(mkpts0_refined, mkpts1_refined),
    #                                 output_path='output/exp_refiner_view.png')
    print("****************** exp refiner done! ******************")

    # 按y坐标排序
    mkpts0_refined = mkpts0_refined[mkpts0_refined[:, 1].argsort()]
    mkpts1_refined = mkpts1_refined[mkpts1_refined[:, 1].argsort()]

    mkpts0_refined_ = mkpts0_refined.copy()
    mkpts1_refined_ = mkpts1_refined.copy()

    img0_color_ = img0_color.copy()
    img1_color_ = img1_color.copy()

    ''' 亚像素细化 '''
    print(f"亚像素化前 mkpts0_refined: {mkpts0_refined_.shape[0]}\n{mkpts0_refined_}")
    refiner = Matching_Subpixel_Refiner(img0_color_, img1_color_, mkpts0_refined_, mkpts1_refined_)
    mkpts0_subpixel, mkpts1_subpixel = refiner.refine_subpixel_cv((5, 5), (-1, -1), 50, 0.001)
    print(f"亚像素化后 mkpts0_subpixel: {mkpts0_subpixel.shape[0]}\n{mkpts0_subpixel}")

    _mkpts0_refined_ = mkpts0_refined.copy()
    _mkpts1_refined_ = mkpts1_refined.copy()
    ''' 亚像素细化可视化 '''
    visualizer = Subpixel_Refiner_View(img0_color_, img1_color_, _mkpts0_refined_, _mkpts1_refined_, mkpts0_subpixel, mkpts1_subpixel)
    result_imgl = visualizer.compare_view_opencv_pro(img0_color_, _mkpts0_refined_, mkpts0_subpixel)
    result_imgr = visualizer.compare_view_opencv_pro(img1_color_, _mkpts1_refined_, mkpts1_subpixel)
    vis_img = np.concatenate([result_imgl, result_imgr], axis=1)
    cv2.namedWindow('compare view opencv pro', 0)
    cv2.resizeWindow('compare view opencv pro', 1500, 1110)
    cv2.imshow("compare view opencv pro", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # visualizer.compare_view_matplotlib(orig_marker='x', sub_marker='o', orig_color='yellow', sub_color='red',
    #                                    line_color='green', point_size=3, outpath='output/exp_subpixel_compare_view_matplotlib.png')
    visualizer.compare_view_matplotlib(outpath='output/exp_subpixel_compare_view_matplotlib.png')


    ''' 亚像素细化评估'''
    evaluator = Subpixel_Refiner_Eval(_mkpts0_refined_, _mkpts1_refined_, mkpts0_subpixel, mkpts1_subpixel)
    evaluator.plot_displacement_histogram(output='output/exp_subpixel_displacement_histogram.png')
    try:
        metrics = evaluator.compute_metrics()
        print("==== 亚像素优化效果评估 ====")
        for k, v in metrics.items():
            print(f"{k:20}: {v:.4f}")

        evaluator.plot_displacement_vectors(img0_color_, img1_color_, scale_factor=20, output='output/exp_subpixel_displacement_vectors.png')

    except ValueError as e:
        print(f"评估失败: {str(e)}")

    print("****************** exp subpixel done! ******************")
