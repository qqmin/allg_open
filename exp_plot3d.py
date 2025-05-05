import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, splprep, splev
import argparse
import time
import logging
import warnings
import os
import cv2


def save_3d_CV(mkpts0, mkpts1, output=None):
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

    # 校正特征点坐标 (NOTE: 输入的关键点 匹配优化过程已经校正 因此不需要再次输入畸变参数校正)
    mkpts0_rectified = cv2.undistortPoints(mkpts0, K1, None, P=K1)
    mkpts1_rectified = cv2.undistortPoints(mkpts1, K2, None, P=K2)

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

    if output is not None:
        np.savetxt(output, points_3d, fmt='%.6f', delimiter='\t')

    return points_3d


def save_3d_uv2XYZ(mkpts0, mkpts1, output=None):
    mkpts3D_X = []
    mkpts3D_Y = []
    mkpts3D_Z = []

    assert len(mkpts0) == len(mkpts1), f'mkpts 关键点维度不一'
    for (lx, ly), (rx, ry) in zip(mkpts0, mkpts1):
        _, mX, mY, mZ = uv2XYZ(lx, ly, rx, ry)
        mkpts3D_X.append(mX)
        mkpts3D_Y.append(mY)
        mkpts3D_Z.append(mZ)

    # 将计算出的3D坐标保存至txt文件
    # points_3d = [(mkpts3D_X[i], mkpts3D_Y[i], mkpts3D_Z[i]) for i in range(len(mkpts3D_X))]
    # points_3d = np.array(points_3d).reshape(-1, 3)
    points_3d = np.column_stack((mkpts3D_X, mkpts3D_Y, mkpts3D_Z))

    if output is not None:
        np.savetxt(output, points_3d, fmt='%.6f', delimiter='\t')

    return points_3d


def uv2XYZ(lx, ly, rx, ry):
    from calib_para.stereo_config import stereo_camera, get_triangulate_points

    K_left = stereo_camera().K_left
    R_left = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T_left = np.array([[0], [0], [0]])
    K_right = stereo_camera().K_right
    R_right = stereo_camera().R
    T_right = stereo_camera().T

    mLeft = np.hstack([R_left, T_left])
    mLeftM = np.dot(K_left, mLeft)
    mRight = np.hstack([R_right, T_right])
    mRightM = np.dot(K_right, mRight)
    # print(f'mLeft: {mLeft}, mLeftM: {mLeftM}')
    # 最小二乘法A矩阵
    A = np.zeros(shape=(4, 3))
    for i in range(0, 3):
        A[0][i] = lx * mLeftM[2, i] - mLeftM[0][i]
    for i in range(0, 3):
        A[1][i] = ly * mLeftM[2][i] - mLeftM[1][i]
    for i in range(0, 3):
        A[2][i] = rx * mRightM[2][i] - mRightM[0][i]
    for i in range(0, 3):
        A[3][i] = ry * mRightM[2][i] - mRightM[1][i]
    # 最小二乘法B矩阵
    B = np.zeros(shape=(4, 1))
    for i in range(0, 2):
        B[i][0] = mLeftM[i][3] - lx * mLeftM[2][3]
    for i in range(2, 4):
        B[i][0] = mRightM[i - 2][3] - rx * mRightM[2][3]

    XYZ = np.zeros(shape=(3, 1))

    # world_3d = np.matmul(np.matrix(A).I, B)  # 其他代码摘抄 不知道什么意思
    # print(world_3d)

    cv2.solve(A, B, XYZ, cv2.DECOMP_SVD)

    [X], [Y], [Z] = [XYZ[i] for i in range(0, len(XYZ))]
    # X, Y, Z = [np.squeeze(XYZ[i]) for i in range(0, len(XYZ))]

    # print(f'XYZ: {XYZ.flatten()}')
    # print(f'X:{X}\tY:{Y}\tZ:{Z}')
    return XYZ, X, Y, Z


def load_3d_data(input=None):
    # "input.txt" 中的数据格式为: X Y Z (以空格分隔)
    data = np.loadtxt(input)
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]

    # 构建网格 进行数据插值 曲面拟合
    # 为了生成连续曲面 首先构造 X Y 的网格
    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # 使用 cubic 插值方法计算网格上对应的 Z 值
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')

    return X, Y, Z, xi, yi, zi


def plot_3d_meshgrid(X, Y, Z, xi, yi, zi, output=None):
    """ 绘制 3D 曲面拟合图 """
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax1 = fig.add_subplot(121, projection='3d')

    scatter = ax1.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o', s=20)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Scatter Plot')

    cbar1 = fig.colorbar(scatter, ax=ax1, shrink=0.5)
    cbar1.ax.tick_params(labelsize=10)

    ax2 = fig.add_subplot(122, projection='3d')

    surf = ax2.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)

    ax2.scatter(X, Y, Z, c='red', s=10)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('3D Surface Fitting Plot')
    cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.5)
    cbar2.ax.tick_params(labelsize=10)

    plt.tight_layout()
    if output is not None:
        plt.savefig(output, bbox_inches='tight', dpi=300)
    plt.show()


def plot_3d_original(X, Y, Z, output=None, demo=True):

    if not demo:
        mpl.use('Agg')  # 使用非交互式后端 加快批量保存速度
    else:
        pass

    fig = plt.figure(figsize=(9, 6), dpi=200)  # 分辨率 = figsize × dpi
    # 将projection='3d'传递给 Figure.add_subplot 来创建3D轴
    ax = plt.axes(projection='3d')

    # 设置坐标轴范围
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    # ax.set_zlim(980, 1100)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # 绘制图形
    ax.plot(X, Y, Z, label='Vibration Mode Fitting',
            color='blue',
            # marker='o',
            # linestyle='dashed',
            linewidth=2,
            markersize=3
            )

    # 添加散点 用于突出曲线上的特定点 这里选取曲线的起始点和终点
    start_point = ax.scatter(X[0], Y[0], Z[0], c='blue', marker='o', s=10)  # s为scatter size
    end_point = ax.scatter(X[-1], Y[-1], Z[-1], c='red', marker='o', s=10)
    # 使用颜色映射 例如根据 z 值来着色
    scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=10)
    cbar = fig.colorbar(scatter, shrink=0.5)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    cbar.set_label('Z (mm)')
    # cbar.set_label('Z (mm)', rotation=270, labelpad=15)  # labelpad 为标签与轴的距离

    # 在 XY 平面上绘制投影
    z_offset = np.max(Z) + 60  # 可根据需求调整偏移量 使投影与曲面区分开
    # 使用参数 zs 将 Z 坐标固定为 z_offset 同时 zdir 指定投影方向为 'Z'
    ax.plot(X, Y, zs=z_offset, zdir='Z', label="XY Projection",
            color='red', linestyle='--', linewidth=2)
    # 为了更好地显示投影 可调整 Z 轴范围
    ax.set_zlim(np.min(Z) - 60, z_offset)
    scatter = ax.scatter(X, Y, zs=z_offset, c=Z, cmap='viridis', s=10)

    # 标注坐标值
    annotate_xyz = False  # 是否标注坐标值
    if annotate_xyz:
        points = [(X[i], Y[i], Z[i]) for i in range(0, len(X))]
        for point in points:
            # ax.text(point[0], point[1], point[2], f'({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})')
            # 简单地使用固定偏移量
            offset = np.array([10, 0, 0])
            ax.text(point[0]+offset[0], point[1]+offset[1], point[2]+offset[2],
                    f'({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})',
                    color='red',
                    size=8,
                    zorder=1,                       # 设置标注的zorder 确保它在曲线之上
                    # horizontalalignment='center',  # 设置水平对齐方式
                    verticalalignment='bottom',     # 设置垂直对齐方式
                    alpha=0.5                       # 设置透明度
                    )

    ax.legend(loc='upper right')  # 显示图例
    plt.title('3D Vibration Mode')  # 设置标题
    plt.tight_layout()
    # 保存设置参数:
    # transparent=True: 会让坐标轴以及图像补丁(alpha为0的位置)都变为透明
    # bbox_inches='tight' 和 pad_inches=0.0: 保存图像时删除图像的白边
    if output is not None:
        fig.savefig(output)
    if demo:
        plt.show()
    # plt.cla()
    plt.close(fig)


def plot_3d_fit(X, Y, Z, fit=True, output=None):
    if fit:
        # 使用样条插值进行参数化拟合
        # s=0 表示通过所有数据点 若数据有噪声可适当调大s值来平滑曲线
        tck, u = splprep([X, Y, Z], s=0.1, k=3)  # k=3为三次样条

        # 生成更多的参数点以获得平滑曲线
        u_fine = np.linspace(0, 1, 100)  # 插值点数 越多曲线越平滑
        X_fit, Y_fit, Z_fit = splev(u_fine, tck)
    else:
        X_fit, Y_fit, Z_fit = X, Y, Z

    fig = plt.figure(figsize=(12, 6), dpi=200)

    ''' 原始数据 绘图 '''
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.set_xlim(-250, 250)
    ax1.set_ylim(-250, 250)
    # ax1.set_zlim(980, 1100)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)'

    ax1.plot(X, Y, Z, label='parametric curve',
             color='blue',
             # marker='o',
             # linestyle='dashed',
             linewidth=2,
             markersize=12
             )

    start_point = ax1.scatter(X[0], Y[0], Z[0], c='blue', marker='o', s=10)
    end_point = ax1.scatter(X[-1], Y[-1], Z[-1], c='red', marker='o', s=10)

    scatter = ax1.scatter(X, Y, Z, c=Z, cmap='viridis', s=10)
    cbar = plt.colorbar(scatter, shrink=0.5)
    cbar.set_label('Z (mm)')
    # cbar.set_label('Z (mm)', rotation=270, labelpad=15)

    z_offset = np.max(Z) + 60
    ax1.plot(X, Y, zs=z_offset, zdir='Z', label="XY Projection",
             color='red', linestyle='--', linewidth=2)
    ax1.set_zlim(np.min(Z) - 60, z_offset)
    scatter = ax1.scatter(X, Y, zs=z_offset, c=Z, cmap='viridis', s=10)

    # 标注坐标值
    annotate_xyz = True  # 是否标注坐标值
    if annotate_xyz:
        points = [(X[i], Y[i], Z[i]) for i in range(0, len(X))]
        for point in points:
            # ax.text(point[0], point[1], point[2], f'({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})')
            # 简单地使用固定偏移量
            offset = np.array([10, 0, 0])
            ax1.text(point[0]+offset[0], point[1]+offset[1], point[2]+offset[2],
                     f'({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})',
                     color='red',
                     size=8,
                     zorder=1,                       # 设置标注的zorder 确保它在曲线之上
                     # horizontalalignment='center',  # 设置水平对齐方式
                     verticalalignment='bottom',     # 设置垂直对齐方式
                     alpha=0.5                       # 设置透明度
                     )
    ax1.legend(loc='upper right')

    ''' 拟合 绘图 '''
    ax2 = fig.add_subplot(122, projection='3d')

    ax2.set_xlim(-250, 250)
    ax2.set_ylim(-250, 250)
    # ax2.set_zlim(980, 1100)

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')


    ax2.plot(X_fit, Y_fit, Z_fit, label='parametric curve',
             color='blue',
             # marker='o',
             # linestyle='dashed',
             linewidth=2,
             markersize=12
             )

    z_offset = np.max(Z_fit) + 60
    ax2.plot(X_fit, Y_fit, zs=z_offset, zdir='Z_fit', label="XY Projection",
             color='red', linestyle='--', linewidth=2)
    ax2.set_zlim(np.min(Z_fit) - 60, z_offset)
    # scatter = ax2.scatter(X_fit, Y_fit, zs=z_offset, c=Z_fit, cmap='viridis', s=5)

    ax2.legend(loc='upper right')

    plt.title('3D Parametric Curve')
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    from exp_model import args_parse_mqq, initial_matching
    args, _ = args_parse_mqq(description='demo matching')  # args

    ''' 加载模型 '''
    img0_color, img1_color, mkpts0, mkpts1, mconf, color = initial_matching(args)
    print(f'mkpts0: {mkpts0.shape}, mkpts1: {mkpts1.shape}')

    mkpts0_ = mkpts0.copy()
    mkpts1_ = mkpts1.copy()

    ''' 匹配优化 '''
    from exp_refiner import MatcherRefiner
    refiner = MatcherRefiner()
    mkpts0_refined, mkpts1_refined = refiner.refine_matches(
        mkpts0_, mkpts1_, y_thresh=50, ransac_thresh=7, ransac_confidence=0.995,
        max_iters=2000, epipolar_thresh=20, depth_z_thresh=50)
    print(f"mkpts0_refined: {mkpts0_refined.shape[0]}, mkpts1_refined: {mkpts1_refined.shape[0]}")
    print(f"--> 平均重投影误差: {refiner.compute_reprojection_error(mkpts0_refined, mkpts1_refined):.2f} 像素")

    # 按y坐标排序
    mkpts0_refined = mkpts0_refined[mkpts0_refined[:, 1].argsort()]
    mkpts1_refined = mkpts1_refined[mkpts1_refined[:, 1].argsort()]
    mkpts0_refined_ = mkpts0_refined.copy()
    mkpts1_refined_ = mkpts1_refined.copy()

    ''' 亚像素细化'''
    from exp_subpixel import Matching_Subpixel_Refiner, Subpixel_Refiner_Eval
    refiner = Matching_Subpixel_Refiner(img0_color, img1_color, mkpts0_refined_, mkpts1_refined_)
    mkpts0_subpixel, mkpts1_subpixel = refiner.refine_subpixel_cv((5, 5), (-1, -1), 50, 0.001)

    _mkpts0_refined_ = mkpts0_refined.copy()
    _mkpts1_refined_ = mkpts1_refined.copy()
    evaluator = Subpixel_Refiner_Eval(_mkpts0_refined_, _mkpts1_refined_, mkpts0_subpixel, mkpts1_subpixel)
    metrics = evaluator.compute_metrics()
    # print("==== 亚像素优化效果评估 ====")
    # for k, v in metrics.items():
    #     print(f"{k:20}: {v:.4f}")

    mkpts0_subpixel_ = mkpts0_subpixel.copy()
    mkpts1_subpixel_ = mkpts1_subpixel.copy()

    ''' plot 3d '''
    use_CV = True

    output_dir = 'output_tmp/threeD/demo/'
    os.makedirs(output_dir, exist_ok=True)

    if use_CV:
        plot_3d_CV = save_3d_CV(mkpts0_subpixel_, mkpts1_subpixel_, output=f'{output_dir}/plot_3d_CV.txt')
        # print(f'plot_3d_CV: {plot_3d_CV.shape}\n{plot_3d_CV}')

        X_CV, Y_CV, Z_CV, xi_CV, yi_CV, zi_CV = load_3d_data(f'{output_dir}/plot_3d_CV.txt')

        plot_3d_original(X_CV, Y_CV, Z_CV, output=f'{output_dir}/plot_3d_CV_original.png')

        # plot_3d_meshgrid(X_CV, Y_CV, Z_CV, xi_CV, yi_CV, zi_CV, output=None)

        plot_3d_fit(X_CV, Y_CV, Z_CV, output=f'{output_dir}/plot_3d_CV_fit.png')

    else:
        plot_3d_uv2XYZ = save_3d_uv2XYZ(mkpts0_subpixel_, mkpts1_subpixel_, output=f'{output_dir}/plot_3d_uv2XYZ.txt')

        X_uv2XYZ, Y_uv2XYZ, Z_uv2XYZ, xi_uv2XYZ, yi_uv2XYZ, zi_uv2XYZ = load_3d_data(f'{output_dir}/plot_3d_uv2XYZ.txt')

        plot_3d_original(X_uv2XYZ, Y_uv2XYZ, Z_uv2XYZ, output=f'{output_dir}/plot_3d_uv2XYZ_original.png')

        # plot_3d_meshgrid(X_uv2XYZ, Y_uv2XYZ, Z_uv2XYZ, xi_uv2XYZ, yi_uv2XYZ, zi_uv2XYZ, output=None)

        plot_3d_fit(X_uv2XYZ, Y_uv2XYZ, Z_uv2XYZ, output=f'{output_dir}/plot_3d_uv2XYZ_fit.png')

    print("****************** exp plot3d done! ******************")
