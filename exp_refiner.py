import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from calib_para.stereo_config import stereo_camera
from exp_model import camera_considered


class MatcherRefiner:

    def __init__(self):
        self.K1 = stereo_camera().K_left
        self.K2 = stereo_camera().K_right
        self.dist1 = stereo_camera().dist_left
        self.dist2 = stereo_camera().dist_right
        self.R = stereo_camera().R
        self.T = stereo_camera().T.reshape(3, 1)
        # 计算基础矩阵F和投影矩阵
        self._compute_fundamental_matrix()
        self._create_projection_matrices()

    def _compute_fundamental_matrix(self):

        self.T_skew = np.array([[0, -self.T[2, 0], self.T[1, 0]],
                                [self.T[2, 0], 0, -self.T[0, 0]],
                                [-self.T[1, 0], self.T[0, 0], 0]
                                ], dtype=np.float32)

        self.E = self.T_skew @ self.R  # E = [t]_× * R
        self.F = np.linalg.inv(self.K2).T @ self.E @ np.linalg.inv(self.K1)  # F = K2^{-T} * E * K1^{-1}
        # print("利用相机参数计算得到基础矩阵F:\n", self.F)

    def _create_projection_matrices(self):
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])  # 左相机投影矩阵 P1 = K1 * [I | 0]
        self.P2 = self.K2 @ np.hstack([self.R, self.T])  # 右相机投影矩阵 P2 = K2 * [R | T]
        # print("利用相机参数计算投影矩阵P1:\n", self.P1)
        # print("利用相机参数计算投影矩阵P2:\n", self.P2)

    def refine_matches(self, mkpts0, mkpts1):

        assert mkpts0 is not None and mkpts1 is not None, "左右特征点不能为空"
        assert len(mkpts0) == len(mkpts1), "左右特征点数量不一致"  # mkpts0, mkpts1: (Nx2)
        mkpts0 = np.asarray(mkpts0, dtype=np.float32)
        mkpts1 = np.asarray(mkpts1, dtype=np.float32)

        is_rectified = False  # 是否进行立体校正
        cam_considered_dict = camera_considered(mkpts0, mkpts1)
        is_rectified = cam_considered_dict['is_rectified']
        print(f'is_rectified: {is_rectified}')

        mkpts0_rectified = cam_considered_dict['mkpts0_rectified']
        mkpts1_rectified = cam_considered_dict['mkpts1_rectified']
        mkpts0_ = mkpts0_rectified.copy() if is_rectified else mkpts0.copy()
        mkpts1_ = mkpts1_rectified.copy() if is_rectified else mkpts1.copy()
        mkpts0_ = mkpts0_.reshape(-1, 2)
        mkpts1_ = mkpts1_.reshape(-1, 2)
        # print(f'mkpts0_: {mkpts0_.shape}\n{mkpts0_}\n\nmkpts1_: {mkpts1_.shape}\n{mkpts1_}')

        ''' RANSAC基础矩阵验证 '''
        # mkpts0_, mkpts1_, _ = self._ransac_filter(mkpts0_, mkpts1_, ransac_thresh, max_iters, ransac_confidence)
        return mkpts0_, mkpts1_


    def _ransac_filter(self, mkpts0, mkpts1, ransac_threshold, confidence, max_iters):
        """ RANSAC基础矩阵验证 """
        # 使用LMEDS算法提升对噪声的鲁棒性
        if len(mkpts0) >= 7:  # 满足最小样本要求

            F, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC, ransac_threshold, confidence, max_iters)
            # F, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransac_threshold, confidence, max_iters)
            print("--> 阶段2: ransac_filter: %d/%d" % (np.sum(mask), len(mask)))
            # print("findFundamentalMat 函数得到基础矩阵F:\n", F)

            return mkpts0[mask.ravel() == 1], mkpts1[mask.ravel() == 1], F
        print("不满足最小样本要求, RANSAC过滤跳过")

        return mkpts0, mkpts1, None


class VisualizerRefiner:
    def __init__(self, img_left: np.ndarray, img_right: np.ndarray):
        """ 双目匹配结果优化前后可视化  img_left img_right: (H,W,3) BGR格式 """
        assert img_left is not None and img_right is not None, "无法读取图像"
        self.img_left = self._preprocess_image(img_left)
        self.img_right = self._preprocess_image(img_right)
        self._validate_images()
        self._set_academic_style()

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """ 图像预处理 自动调整过亮/过暗图像 """
        assert isinstance(img, np.ndarray), "输入图像必须是 numpy 数组"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return img

    def _validate_images(self):
        """ 验证图像尺寸和类型 """
        assert self.img_left.shape == self.img_right.shape, "左右图像尺寸不一致"
        assert self.img_left.ndim == 3 and self.img_right.ndim == 3, "需输入彩色图像"

    def _set_academic_style(self):
        """ 设置学术绘图样式 """
        rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.autolayout': True,
            'lines.linewidth': 0.8,
            'lines.markersize': 3,
        })

    def visualize_comparison(self, method, original_matches, refined_matches, output_path=None):
        """
        method: 可视化方式 'matplotlib' 或 'opencv'
        original_matches: 原始匹配点对 (mkpts0, mkpts1)  Tuple[np.ndarray, np.ndarray]
        refined_matches: 优化匹配点对 (mkpts0, mkpts1)  Tuple[np.ndarray, np.ndarray]
        output_path: 可视化结果保存路径
        """
        if method.lower() == 'opencv':
            self._visualize_with_opencv(original_matches, refined_matches)
        elif method.lower() == 'matplotlib':
            self._visualize_with_matplotlib(original_matches, refined_matches, output_path)
        else:
            raise ValueError("不支持的显示方法 请选择'matplotlib'或'opencv'")

    def _visualize_with_opencv(self, original_matches, refined_matches):
        print('visualizing_with_opencv...')
        img_left = self.img_left.copy()
        img_right = self.img_right.copy()
        vis_img = np.hstack([img_left, img_right])
        overlay = vis_img.copy()

        ''' 绘制原始匹配 (半透明) '''
        mkpts0, mkpts1 = original_matches
        w = self.img_left.shape[1]
        for idx in range(len(mkpts0)):
            x1, y1 = map(int, mkpts0[idx])
            x2, y2 = map(int, mkpts1[idx])
            cv2.line(overlay, (x1, y1), (x2 + w, y2), (0, 0, 255), 2, lineType=cv2.LINE_AA)  # BGR红色
            cv2.circle(overlay, (x1, y1), 3, (255, 0, 0), -1)
            cv2.circle(overlay, (x2 + w, y2), 3, (255, 0, 0), -1)

        ''' 绘制优化匹配 (不透明) '''
        mkpts0, mkpts1 = refined_matches
        for idx in range(len(mkpts0)):
            x1, y1 = map(int, mkpts0[idx])
            x2, y2 = map(int, mkpts1[idx])
            cv2.line(vis_img, (x1, y1), (x2 + w, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)  # BGR绿色  黄色
            cv2.circle(vis_img, (x1, y1), 3, (255, 0, 0), -1)
            cv2.circle(vis_img, (x2 + w, y2), 3, (255, 0, 0), -1)

        # 叠加图像
        cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0, vis_img)

        h, w = vis_img.shape[:2]
        cv2.putText(vis_img, "Red: Original", (w-400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(vis_img, "Yellow: Refined", (w-400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        cv2.namedWindow('Match Comparison', 0)
        cv2.resizeWindow('Match Comparison', 1500, 1110)
        cv2.imshow("Match Comparison", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _visualize_with_matplotlib(self, original_matches, refined_matches, output_path):
        print('visualizing_with_matplotlib...')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # figsize=(20, 10)
        plt.suptitle("Stereo Matching Refinement Comparison", y=0.98, fontsize=14)
        ''' 绘制原始匹配 '''
        mkpts0, mkpts1 = original_matches
        w = self.img_left.shape[1]
        ax1.imshow(np.hstack([self.img_left, self.img_right]))
        ax1.axis('off')
        ax1.set_title(f"{'Original Matches'} ({len(mkpts0)} pairs)", fontsize=12)
        # 绘制匹配线
        for idx in range(len(mkpts0)):
            x1, y1 = mkpts0[idx]
            x2, y2 = mkpts1[idx]
            ax1.plot([x1, x2 + w], [y1, y2], 'r-', linewidth=0.3)
            ax1.plot(x1, y1, 'b.', markersize=0.3)
            ax1.plot(x2 + w, y2, 'b.', markersize=0.3)
        # 添加统计信息
        stats_text = (f"Matches: {len(mkpts0)}")
        bbox = dict(facecolor='white', alpha=0.8, edgecolor='none')
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top', bbox=bbox)

        ''' 绘制优化匹配 '''
        mkpts0, mkpts1 = refined_matches
        ax2.imshow(np.hstack([self.img_left, self.img_right]))
        ax2.axis('off')
        ax2.set_title(f"{'Refined Matches'} ({len(mkpts0)} pairs)", fontsize=12)
        # 绘制匹配线
        for idx in range(len(mkpts0)):
            x1, y1 = mkpts0[idx]
            x2, y2 = mkpts1[idx]
            ax2.plot([x1, x2 + w], [y1, y2], 'g-', linewidth=0.3)
            ax2.plot(x1, y1, 'bo', markersize=0.3)
            ax2.plot(x2 + w, y2, 'bo', markersize=0.3)
        # 添加统计信息
        stats_text = (f"Matches: {len(mkpts0)}")
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, va='top', bbox=bbox)

        plt.tight_layout()

        if output_path is not None:
            # plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.savefig(output_path)
            print(f"Figure saved to: {output_path}")
        plt.show()
        plt.close(fig)
        return fig


if __name__ == "__main__":

    from exp_model import args_parse_mqq, initial_matching
    args, _ = args_parse_mqq(description='demo matching')  # args

    # img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args, 'input/L2.bmp', 'input/R2.bmp')
    img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args)
    print(f'mkpts0: {mkpts0.shape}\nmkpts1: {mkpts1.shape}')

    # 初始化优化器
    refiner = MatcherRefiner()
    mkpts0_refined, mkpts1_refined = refiner.refine_matches(mkpts0, mkpts1, y_thresh=50, ransac_thresh=7,
                                                            ransac_confidence=0.995, max_iters=2000,
                                                            epipolar_thresh=20, depth_z_thresh=50)

    # 评估结果
    print(f"优化匹配数/原始匹配数: {len(mkpts0_refined)}/{len(mkpts0)}")
    print(f"平均重投影误差: {refiner.compute_reprojection_error(mkpts0_refined, mkpts1_refined):.2f} 像素")

    # 可视化结果
    visualizer = VisualizerRefiner(img0_color, img1_color)
    visualizer.visualize_comparison(method='opencv', original_matches=(mkpts0, mkpts1),
                                    refined_matches=(mkpts0_refined, mkpts1_refined))
    visualizer.visualize_comparison(method='matplotlib', original_matches=(mkpts0, mkpts1),
                                    refined_matches=(mkpts0_refined, mkpts1_refined),
                                    output_path='output/exp_refiner_view.png')

    print("****************** exp refiner done! ******************")
