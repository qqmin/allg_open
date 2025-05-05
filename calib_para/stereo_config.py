'''
camera_id.Width.SetValue(1440)
camera_id.Height.SetValue(2000)
camera_id.OffsetX.SetValue(1920)
camera_id.OffsetY.SetValue(520)
'''
import cv2
import numpy as np

np.set_printoptions(suppress=True)


calib_para_path = './calib_para/20250327/calib_para_light.txt'

with open(calib_para_path, 'r') as file:
    assert len(file.readlines()) == 34, f'相机标定文件参数不对'

calib_data = np.loadtxt(calib_para_path, delimiter=',')
# print('calib_data:\n', calib_data)

height, width = int(calib_data[0]), int(calib_data[1])
# print(f'height:{height}, width:{width}')

# 左
fx_l, fy_l, tao_l, cx_l, cy_l = [calib_data[i] for i in range(2, 7)]
k1_l, k2_l, p1_l, p2_l, k3_l = [calib_data[i] for i in range(7, 12)]
# print(f'fx_l:{fx_l}, fy_l:{fy_l}, tao_l:{tao_l}, cx_l:{cx_l}, cy_l:{cy_l}')
# print(f'k1_l:{k1_l}, k2_l:{k2_l}, p1_l:{p1_l}, p2_l:{p2_l}, k3_l:{k3_l}')

# 右
fx_r, fy_r, tao_r, cx_r, cy_r = [calib_data[i] for i in range(12, 17)]
k1_r, k2_r, p1_r, p2_r, k3_r = [calib_data[i] for i in range(17, 22)]
# print(f'fx_r:{fx_r}, fy_r:{fy_r}, tao_r:{tao_r}, cx_r:{cx_r}, cy_r:{cy_r}')
# print(f'k1_r:{k1_r}, k2_r:{k2_r}, p1_r:{p1_r}, p2_r:{p2_r}, k3_r:{k3_r}')

# 双目
R_r2l = np.array([calib_data[i] for i in range(22, 31)]).reshape(3, 3).T
T_r2l = np.array([[calib_data[i] for i in range(31, 34)]]).T
# print(f'R_r2l:\n{R_r2l}\nT_r2l:\n{T_r2l}')


class stereo_camera(object):
    def __init__(self):
        self.height, self.width = height, width
        self.K_left = np.array([[fx_l, tao_l, cx_l],
                                [0,     fy_l, cy_l],
                                [0,        0,    1]])
        self.K_right = np.array([[fx_r, tao_r, cx_r],
                                 [0,     fy_r, cy_r],
                                 [0,        0,    1]])

        self.dist_left = np.array([[k1_l, k2_l, p1_l, p2_l, k3_l]])
        self.dist_right = np.array([[k1_r, k2_r, p1_r, p2_r, k3_r]])

        self.R = R_r2l
        self.T = T_r2l


def new_camera_matrix(image, camera_matrix, dist_coeff):

    height, width = image.shape[:2]
    new_cam_mat, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (width, height), alpha=1)
    x, y, w, h = valid_roi

    # 使用默认参数的时候主要控制参数是利用 newCameraMatrix 来完成, newCameraMatrix 一般由 getOptimalNewCameraMatrix() 函数得到
    undist_img = cv2.undistort(image, camera_matrix, dist_coeff, None, new_cam_mat)

    # cv2.imshow('undistorted image', undist_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return new_cam_mat, undist_img


def get_rectify_img(image1, image2, stereo_config):

    K_left = stereo_config.K_left
    K_right = stereo_config.K_right
    dist_left = stereo_config.dist_left
    dist_right = stereo_config.dist_right
    R = stereo_config.R
    T = stereo_config.T

    assert image1.shape == image2.shape, f'image1.shape =?= image2.shape'
    height, width = image1.shape[:2]

    # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, (width, height), R, T, alpha=0)
    # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, (width, height), R, T, flags=0, alpha=1)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, (width, height), R, T, flags=0, alpha=0)
    '''

                                                   [f  0  cx1  0]           [f  0  cx2  T_x*f]          [1  0       0           -cx1]
    R1 =        R2 =                          P1 = |0  f   cy  0|      P2 = |0  f   cy      0|      Q = |0  1       0            -cy|
                                                   [0  0    1  0]           [0  0    1      0]          |0  0       0              f|
                                                                                                        [0  0  -1/T_x  (cx1-cx2)/T_x]
    '''
    map1x, map1y = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, (width, height), cv2.CV_32FC1)
    # map1x, map1y = cv2.initUndistortRectifyMap(K_left, dist_left, R1, K_left, (width, height), cv2.CV_32FC1)  # 只使用R1, 不使用P1
    map2x, map2y = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, (width, height), cv2.CV_32FC1)

    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)  # cv2.INTER_LINEAR cv2.INTER_NEAREST
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return Q, rectifyed_img1, rectifyed_img2


def get_triangulate_points(kpts_l, kpts_r, stereo_config):

    # .astype(np.float32)?
    K_left = stereo_config.K_left
    K_right = stereo_config.K_right
    dist_left = stereo_config.dist_left
    dist_right = stereo_config.dist_right
    R = stereo_config.R
    T = stereo_config.T
    # print(f'K_left:\n{K_left}\nK_right:\n{K_right}\ndist_left:\n{dist_left}\ndist_right:\n{dist_right}\nR:\n{R}\nT:\n{T}')
    # print(f'shape: K_left:{K_left.shape} K_right:{K_right.shape} dist_left:{dist_left.shape} dist_right:{dist_right.shape} R:{R.shape} T:{T.shape}')

    K_left_inv = np.linalg.inv(K_left)
    K_right_inv = np.linalg.inv(K_right)

    kpts_l = np.ascontiguousarray(kpts_l)
    kpts_r = np.ascontiguousarray(kpts_r)
    # 畸变校正
    '''
    x" <-- (u - cx) / fx
    y" <-- (v - cy) / fy
    (x', y') = undistort(x", y", distCoeffs)
    [X Y W]^T <-- R * [x' y' 1]^T
    x <-- X / W
    y <-- Y / W
    u' <-- xf'_x + c'x
    v' <-- yf'_y + c'y
    '''
    print(f'kpts_l: {kpts_l.shape}\n{kpts_l}')
    print(f'kpts_r: {kpts_r.shape}\n{kpts_r}')   # shape: (n, 2)
    # kpts_l is a (N, 1, 2) float32 from cornerSubPix
    # undist_left = cv2.undistortPoints(kpts_l, K_left, dist_left)
    # undist_right = cv2.undistortPoints(kpts_r, K_right, dist_right)
    undist_left = cv2.undistortPoints(kpts_l, K_left, dist_left, P=K_left)
    undist_right = cv2.undistortPoints(kpts_r, K_right, dist_right, P=K_right)
    # print(f'undist_right(before):\n{undist_right}\n{undist_right.shape}')   # shape: (n, 1, 2)
    # undistorted_points = cv2.undistortPoints(np.expand_dims(kpts_l, axis=1), K_left, dist_left)
    # print(undistorted_points[:, 0, :])

    undist_left = np.array(undist_left).squeeze()   # shape: (n, 2)
    undist_right = np.array(undist_right).squeeze()

    # undist_left = np.array(undist_left[0]).transpose()
    # undist_right = np.array(undist_right[0]).transpose()
    # print(f'undist_right(after):\n{undist_right}\n{undist_right.shape}')   # shape: (n, 2)

    # kpts_l_homogeneous = np.hstack((undist_left, np.ones((undist_left.shape[0], 1))))
    kpts_l_homogeneous = np.hstack((undist_left, np.ones((undist_left.shape[0], 1)))).astype(np.float32)
    # kpts_r_homogeneous = np.hstack((undist_right, np.ones((undist_right.shape[0], 1))))
    kpts_r_homogeneous = np.hstack((undist_right, np.ones((undist_right.shape[0], 1)))).astype(np.float32)
    print(f'kpts_l_homogeneous(undistorted): {kpts_l_homogeneous.shape}\n{kpts_l_homogeneous}')
    print(f'kpts_r_homogeneous(undistorted): {kpts_r_homogeneous.shape}\n{kpts_r_homogeneous}')   # shape: (n, 3)

    # _, _, P1, P2, _, _, _ = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, (width, height), R, T, flags=0, alpha=0)
    # print(f'projection_matrix(P1):\n{P1.astype(np.float32)}\nprojection_matrix(P2):\n{P2.astype(np.float32)}')

    # 投影矩阵定义为 P = KT (矩阵乘法)   P = K [R | t]     P1 = K1[I∣0]   P2 = K2[R∣t]
    P_left = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_right = K_right @ np.hstack((R, T.reshape(-1, 1)))
    print(f'projection_matrix(P_left):\n{P_left.astype(np.float32)}\nprojection_matrix(P_right):\n{P_right.astype(np.float32)}')

    # points3D_homogeneous = cv2.triangulatePoints(P_left, P_right, undist_left.T, undist_right.T)
    # points3D_homogeneous = cv2.triangulatePoints(P_left, P_right, kpts_l_homogeneous.T, kpts_r_homogeneous.T)
    points3D_homogeneous = cv2.triangulatePoints(P_left, P_right, kpts_l_homogeneous[:, :2].T, kpts_r_homogeneous[:, :2].T)
    # points3D_homogeneous = cv2.triangulatePoints(P_left, P_right, kpts_l_homogeneous[:, :2].reshape(-1, 1, 2), kpts_r_homogeneous[:, :2].reshape(-1, 1, 2))  # 维度调整为 (npoints, 1, 2)

    # points_3d = points3D_homogeneous[:3, :] / points3D_homogeneous[3, :]
    points_3d = cv2.convertPointsFromHomogeneous(points3D_homogeneous.T)

    '''
    # 验证齐次矩阵
    # check RT_l
    rtmtxl_homo = np.vstack((RT_l, np.array([0, 0, 0, 1])))
    obj_homo = cv2.convertPointsToHomogeneous(objpoints_l[0]).reshape(63, 4).T
    print("P*RT:\n", np.dot(rtmtxl_homo, obj_homo))
    '''

    # 输出结果
    # print(f'3D Points: {points_3d[:3].T}, {points_3d.squeeze()}')
    print(f'3D Points:\n{points_3d.squeeze()}')

    return points_3d, points_3d.T


def get_triangulate_points_(kpts_l, kpts_r, stereo_config):

    K_left = stereo_config.K_left
    K_right = stereo_config.K_right
    dist_left = stereo_config.dist_left
    dist_right = stereo_config.dist_right
    R = stereo_config.R
    T = stereo_config.T

    kpts_l = np.ascontiguousarray(kpts_l)
    kpts_r = np.ascontiguousarray(kpts_r)

    undist_left = cv2.undistortPoints(kpts_l, K_left, dist_left, P=K_left)
    undist_right = cv2.undistortPoints(kpts_r, K_right, dist_right, P=K_right)

    # 投影矩阵 P1=K1[I∣0] P2=K2[R∣t]
    P1 = np.dot(K_left, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K_right, np.hstack((R, T.reshape(3, 1))))

    # 三角测量
    points3D_homogeneous = cv2.triangulatePoints(P1, P2, undist_left, undist_right)
    points_3d = cv2.convertPointsFromHomogeneous(points3D_homogeneous.T)
    print(f'3D Points:\n{points_3d.squeeze()}')

    return points_3d, points_3d.T


def reproject_points(img, kpts, camera_matrix, dist_coefs):

    img_pts, _ = cv2.projectPoints(kpts, (0, 0, 0), (0, 0, 0), camera_matrix, None)

    for c in kpts:
        cv2.circle(img, tuple(c[0]), 10, (0, 255, 0), 2)

    for c in img_pts.squeeze().astype(np.float32):
        cv2.circle(img, tuple(c), 5, (0, 0, 255), 2)

    cv2.imshow('undistorted corners', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_pts, _ = cv2.projectPoints(kpts, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coefs)

    for c in img_pts.squeeze().astype(np.float32):
        cv2.circle(img, tuple(c), 2, (255, 255, 0), 2)

    cv2.imshow('reprojected corners', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def draw_line(image1, image2):
    # 画线进行立体校正检验
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    line_interval = 50  # 绘制等间距平行线 直线间隔: 50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    return output


def stereo_match_SGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 64,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


def get_depthmap_withQ(disparity_map: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    depth = ( f * baseline) / disp
    depth表示深度图; f表示归一化的焦距, 也就是内参中的fx; baseline是两个相机光心之间的距离, 称作基线距离; disp是视差值
    直接利用opencv中的cv2.reprojectImageTo3D()函数计算深度图
    '''
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    depth_map = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depth_map < 0.0, depth_map > 65535.0))
    depth_map[reset_index] = 0

    return depth_map.astype(np.float32)


'''
def get_depthmap_withconfig(disparity_map: np.ndarray, config: stereo_camera) -> np.ndarray:
    # 根据公式计算深度图
    fb = config.K_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depth_map = np.divide(fb, disparity_map + doffs)
    reset_index = np.where(np.logical_or(depth_map < 0.0, depth_map > 65535.0))
    depth_map[reset_index] = 0
    reset_index2 = np.where(disparity_map < 0.0)
    depth_map[reset_index2] = 0
    return depth_map.astype(np.float32)
'''
