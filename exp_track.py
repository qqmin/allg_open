import os
import cv2
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt


def extract_Nth_lines_from_files(folder_path, Nth_point):
    Nth_lines = {}
    # 获取文件列表并按文件名排序
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # last_line = lines[-1].strip()   # NOTE 就改这里 原本提取最后一行 改成其他行
            Nth_line = lines[int(Nth_point)].strip()

            Nth_lines[filename] = Nth_line

    return Nth_lines


def get_Nth_point_2d(folder_path_l, folder_path_r, Nth_point):

    # 提取每个文件的第N行数据
    result_l = extract_Nth_lines_from_files(folder_path_l, Nth_point)
    result_r = extract_Nth_lines_from_files(folder_path_r, Nth_point)

    # 将数据保存为NumPy数组
    numpy_array_l = np.array(list(result_l.values()))
    numpy_array_r = np.array(list(result_r.values()))

    # 将数组中的空格替换为逗号
    numpy_array_l = np.char.replace(numpy_array_l, ' ', ',')
    numpy_array_r = np.char.replace(numpy_array_r, ' ', ',')

    # 使用literal_eval将字符串列表转换为实际列表
    Nth_xy_lists_l = [ast.literal_eval(f"[{item}]") for item in numpy_array_l]
    Nth_xy_lists_r = [ast.literal_eval(f"[{item}]") for item in numpy_array_r]

    print(f'-------------------\nloading {Nth_point}th point\n-------------------')
    print(f'{Nth_point}th_xy_lists_l: len {len(Nth_xy_lists_l)}\n{Nth_point}th_xy_lists_r: len {len(Nth_xy_lists_r)}')

    return Nth_xy_lists_l, Nth_xy_lists_r


def get_Nth_point_3d(Nth_xy_lists_l, Nth_xy_lists_r):

    assert len(Nth_xy_lists_l) == len(Nth_xy_lists_r), f'数据维度不一'

    from calib_para.stereo_config import stereo_camera

    K1 = stereo_camera().K_left
    K2 = stereo_camera().K_right
    dist1 = stereo_camera().dist_left
    dist2 = stereo_camera().dist_right
    R = stereo_camera().R
    T = stereo_camera().T.reshape(3, 1)

    Nth_xy_lists_l = np.array(Nth_xy_lists_l)
    Nth_xy_lists_r = np.array(Nth_xy_lists_r)

    mkpts0_rectified = cv2.undistortPoints(Nth_xy_lists_l, K1, None, P=K1)
    mkpts1_rectified = cv2.undistortPoints(Nth_xy_lists_r, K2, None, P=K2)

    # 投影矩阵 P1 = K1[I∣0], P2 = K2[R∣t]
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(-1, 1)))
    # print(f'P1:\n{P1}\nP2:\n{P2}')

    # 三角测量
    points3D_homogeneous = cv2.triangulatePoints(P1, P2, mkpts0_rectified, mkpts1_rectified)
    points_3d = cv2.convertPointsFromHomogeneous(points3D_homogeneous.T)
    # print(f'3D Points:\n{points_3d.squeeze()}')
    points_3d = points_3d.squeeze()
    assert points_3d.shape == (len(Nth_xy_lists_l), 3), "points_3d 必须是 (N, 3) 的形状"

    # points_3d, _ = get_triangulate_points(list_of_lists_l, list_of_lists_r, stereo_camera())
    # points_3d, _ = get_triangulate_points_(Nth_xy_lists_l, Nth_xy_lists_r, stereo_camera())   # 改
    # points_3d = points_3d.squeeze()
    points_3d_X = []
    points_3d_Y = []
    points_3d_Z = []
    for idx in range(len(points_3d)):
        # points_3d_X.append(points_3d[idx][0])
        points_3d_X.append(points_3d[idx][0]-8.12)  # 改  归零 -7.92  -4.432  -8.559  -12.69
        points_3d_Y.append(points_3d[idx][1])
        points_3d_Z.append(points_3d[idx][2])
    print(f'points_3d_X: len {len(points_3d_X)}\npoints_3d_Y: len {len(points_3d_Y)}\npoints_3d_Z: len {len(points_3d_Z)}')

    return points_3d_X, points_3d_Y, points_3d_Z


def get_Nth_point_curve(points_3d_X, points_3d_Y, points_3d_Z, output=None):

    assert len(points_3d_X) == len(points_3d_Y) == len(points_3d_Z), f'数据维度不一'

    time = np.linspace(0, 6, 600)  # 第二个参数是图片采集时间 (s) 第三个参数是图片数量
    # 去除直流分量 (消除均值 避免零频分量干扰)
    points_3d_X = points_3d_X - np.mean(points_3d_X)

    fs = 1 / time[1] - time[0]  # 采样频率
    n = len(points_3d_X)
    # 执行FFT并计算幅值谱 (快速傅里叶变换 获得频域幅值)
    amplitudes = np.abs(np.fft.fft(points_3d_X))
    # 生成对应的频率轴
    freqs = np.fft.fftfreq(n, 1/fs)
    # 提取正频率部分 (排除直流分量)
    mask = freqs >= 0
    fft_freq = freqs[mask]
    fft_result = amplitudes[mask]
    # 排除0Hz分量 (索引从1开始)
    fft_freq = fft_freq[1:]
    fft_result = fft_result[1:]
    # 归一化振幅
    fft_result = fft_result * 2 / n
    # 找到最大幅值的索引 (在正频率范围内找到幅值最大的频率点)
    max_idx = np.argmax(fft_result)
    natural_frequency = fft_freq[max_idx]
    print(f'固有频率为: {natural_frequency:.4f} Hz')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), dpi=200)

    # 绘制傅里叶变换后的频谱图
    axes[0, 0].plot(fft_freq, np.abs(fft_result), label='Frequency Spectrum of X')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Frequency Spectrum')
    axes[0, 0].legend()

    axes[1, 0].plot(time, points_3d_X, label='X (mm)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude-X (mm)')
    axes[1, 0].set_title('Vibration X-axis')
    axes[1, 0].legend()

    axes[0, 1].plot(time, points_3d_Y, label='Y (mm)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude-Y (mm)')
    axes[0, 1].set_title('Vibration Y-axis')
    axes[0, 1].legend()
    # 设置坐标轴范围
    # axes[0, 1].set_ylim(225, 227)

    axes[1, 1].plot(time, points_3d_Z, label='Z (mm)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude-Z (mm)')
    axes[1, 1].set_title('Vibration Z-axis')
    axes[1, 1].legend()
    # 设置坐标轴范围
    # axes[1, 1].set_ylim(1090, 1097)

    plt.tight_layout()
    if output is not None:
        plt.savefig(output)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mkpts directory.')
    parser.add_argument('mkpts0_dir', type=str, default='output_tmp/mkpts0', help='Image L directory.')
    parser.add_argument('mkpts1_dir', type=str, default='output_tmp/mkpts1', help='Image R directory.')
    parser.add_argument('Nth_point', type=int, default='-1', help='Nth_point')
    parser.add_argument('output', type=str, default="output_tmp/spectrum_old.png", help="保存图像的路径")
    args = parser.parse_args()

    dir_path = 'output_tmp/track_dir'
    os.makedirs(dir_path, exist_ok=True)

    Nth_xy_lists_l, Nth_xy_lists_r = get_Nth_point_2d(args.mkpts0_dir, args.mkpts1_dir, args.Nth_point)
    points_3d_X, points_3d_Y, points_3d_Z = get_Nth_point_3d(Nth_xy_lists_l, Nth_xy_lists_r)
    get_Nth_point_curve(points_3d_X, points_3d_Y, points_3d_Z, args.output)

    # np.savetxt(f'{dir_path}/lists_{args.Nth_point}_xy_l.txt', np.array(Nth_xy_lists_l).reshape(-1, 2), fmt='%.4f')
    # np.savetxt(f'{dir_path}/lists_{args.Nth_point}_xy_r.txt', np.array(Nth_xy_lists_r).reshape(-1, 2), fmt='%.4f')
    np.savetxt(f'{dir_path}/lists_{args.Nth_point}_X.txt', np.array(points_3d_X), fmt='%.4f')
    np.savetxt(f'{dir_path}/lists_{args.Nth_point}_Y.txt', np.array(points_3d_Y), fmt='%.4f')
    np.savetxt(f'{dir_path}/lists_{args.Nth_point}_Z.txt', np.array(points_3d_Z), fmt='%.4f')
