import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig  # 用于求解广义特征值问题
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, vstack, hstack, identity
from scipy.sparse.linalg import spsolve, eigsh

np.set_printoptions(suppress=True)       # 去掉e表示
np.set_printoptions(precision=4)         # 设置显示小数点后4位

dynamic_conf = {
    'Disp_Tip_Z_0': 20 * 10 ** (-3),     # 悬臂梁末端施加的初始位移 (m)
    'Force_Tip_Z_0': 0,                  # 悬臂梁末端施加的初始力 (N)
    'm_Tip': 0.0,                        # 末端质量块的重量 (kg)
    'J_Tip': 0,                          # 末端质量块的转动惯量 (kg·m**2)

    'Length_X': 475*10 ** -3,            # 悬臂梁长 X 方向 (m)
    'Length_Y': 10*10 ** -3,             # 悬臂梁宽 Y 方向 (m)
    'Length_Z': 2*10 ** -3,              # 悬臂梁厚 Z 方向 (m)

    'Elastic_modulus': 71e9,             # 弹性模量E (Pa)
    'Density': 2800,                     # 材料密度ρ (kg/m**3)
    'Poisson_ratio': 0.33,               # 泊松比

    'NUM_Cell_Node': 201,                # 悬臂梁网格划分: 节点数

    'Damping_Coefficients_M': 0.0001,    # 0.02593;
    'Damping_Coefficients_K': 0.00025,   # 0.01390;

    'Num_Steps': 500,                    # 总的时间步数
    'Total_time': 5,                     # 总的时长 (s)
}


''' 载荷输入 '''
Disp_Tip_Z_0 = dynamic_conf['Disp_Tip_Z_0']     # 悬臂梁末端施加的初始位移
Force_Tip_Z_0 = dynamic_conf['Force_Tip_Z_0']   # 悬臂梁末端施加的初始力
m_Tip = dynamic_conf['m_Tip']                   # 末端质量块的重量
J_Tip = dynamic_conf['J_Tip']                   # 末端质量块的转动惯量
''' 几何参数 '''
Length_X = dynamic_conf['Length_X']         # 悬臂梁长 X 方向 (m)
Length_Y = dynamic_conf['Length_Y']         # 悬臂梁宽 Y 方向 (m)
Length_Z = dynamic_conf['Length_Z']         # 悬臂梁厚 Z 方向 (m)
Ax = Length_Z * Length_Y                    # 横截面积: 矩形 [m^2]
''' 材料参数 '''
Elastic_modulus = dynamic_conf['Elastic_modulus']   # 弹性模量E (Pa)
Poisson_ratio = dynamic_conf['Poisson_ratio']       # 泊松比
Density = dynamic_conf['Density']                   # 材料密度ρ (kg/m**3)
Shear_modulus = Elastic_modulus/(2 * (1+Poisson_ratio))    # 剪切模量: 由弹性模量和泊松比计算

Inertia_moment = np.zeros((3, 2))                          # 杆件各个方向的惯性矩
Inertia_moment[1, 0] == Length_Y * Length_Z ** 3/12        # 沿y轴的惯性矩
Inertia_moment[2, 0] == Length_Y ** 3 * Length_Z/12        # 沿z轴的惯性矩
Inertia_moment[0, 0] == Inertia_moment[1, 1] + Inertia_moment[2, 1]  # 沿x轴的惯性矩: Ip=Ix+Iy
# Inertia_moment: [Ix; Iy; Iz], 均为 2D np.array (column向量)
# Iy = Length_Y * Length_Z**3/12
# Iz = Length_Y**3 * Length_Z/12
# Ix = Iy + Iz
# Inertia_moment = np.array([[Ix], [Iy], [Iz]])
''' 节点与单元定义 '''
NUM_Cell_Node = dynamic_conf['NUM_Cell_Node']             # 悬臂梁网格划分: 节点数
NUM_Cell_Rod = NUM_Cell_Node - 1                          # 悬臂梁网格划分: 单元数
''' 阻尼参数 '''
Damping_Coefficients_M = dynamic_conf['Damping_Coefficients_M']
Damping_Coefficients_K = dynamic_conf['Damping_Coefficients_K']
''' 计算时间参数 '''
Num_Steps = dynamic_conf['Num_Steps']  # 总的时间步数
Total_time = dynamic_conf['Total_time']  # 总的时长


def cumulated_restrained_list(nnj, rl):
    """
    累计约束表函数: 将非约束节点排在前方
    参数:
        nnj (int): 总自由度数
        rl (list or array-like): 每个自由度的约束标志(0表示自由, 1表示约束)
    返回:
        n    : 非约束自由度数 (int)
        nr   : 约束自由度数 (int)
        crl  : 重排序的自由度索引(0-indexed排列)
    """
    kkl = 0  # 非约束节点数量
    crl = [0] * nnj  # # 初始化 crl 列表 长度与节点总数相同
    # 先处理非约束节点(rl == 0) 将它们排在前面
    for jx in range(nnj):
        if rl[jx] == 0:
            kkl += 1
            crl[jx] = kkl - 1  # 0-indexed
    n = kkl
    # 再处理约束节点(rl == 1)
    for jx in range(nnj):
        if rl[jx] == 1:
            kkl += 1
            crl[jx] = kkl - 1
    nr = kkl - n  # 约束节点数量
    return n, nr, crl


# def cumulated_restrained_list(nnj, rl):
#     """
#     根据约束标志重排自由度, 使得自由(rl==0)的自由度排在前面,
#     约束(rl==1)的自由度排在后面
#     参数:
#       nnj: 总自由度数
#       rl: 约束标志列表(0表示自由, 1表示约束)
#     返回:
#       n: 自由自由度数
#       nr: 约束自由度数
#       new_order: 长度为 nnj 的排列列表, 新顺序中自由度在前, 约束在后
#     """
#     free_indices = [i for i in range(nnj) if rl[i] == 0]
#     fix_indices = [i for i in range(nnj) if rl[i] == 1]
#     n = len(free_indices)
#     nr = len(fix_indices)
#     new_order = free_indices + fix_indices
#     return n, nr, new_order


def space_frame_stiffness_matrix_rod(NUM_Cell_Rod):
    """
    杆件刚度矩阵(全局坐标系下)的转换函数
    计算单元坐标系下杆件刚度矩阵, 并转换到结构(全局)坐标系下
    参数:
        NUM_Cell_Rod (int): 当前杆件的单元编号
    返回:
        S_Rod (array-like): 12x12 的杆件刚度矩阵 已转换到结构坐标系
    """
    # 读取当前杆件对应的两个节点编号(假设节点编号已经是0-indexed)
    NUM_Node = Coordinate_Cell_Rod[NUM_Cell_Rod, :]  # 长度为2的数组
    pj = Coordinate_Cell_Node[NUM_Node[0], :]  # 杆件起始节点坐标
    pk = Coordinate_Cell_Node[NUM_Node[1], :]  # 杆件终止节点坐标

    # 构造局部坐标系 计算杆件沿局部 x 轴的方向向量
    xm = pk - pj
    c = np.sqrt(xm[0]**2 + xm[2]**2)  # 计算 xm 在 x-z 平面上的投影长度 用于判断杆件是否接近垂直
    if c > 0.0001:  # 当杆件不接近垂直时 采用倾斜杆的局部 z 轴方向向量
        zm = np.array([-xm[2], 0, xm[0]])
    else:  # 当杆件接近垂直时 直接取全局 z 轴方向
        zm = np.array([0, 0, 1])
    ym = np.cross(zm, xm)  # 计算局部 y 轴: 取 zm 与 xm 的叉乘
    x_unit = xm / np.linalg.norm(xm)  # 归一化各轴向量
    y_unit = ym / np.linalg.norm(ym)
    z_unit = zm / np.linalg.norm(zm)
    R = np.vstack([x_unit, y_unit, z_unit])  # 构造从全局坐标系到杆件局部坐标系的旋转矩阵 R, 每一行为一个单位向量
    # 构造12x12坐标变换矩阵 RT
    RT = np.zeros((12, 12))  # 构造 12x12 的坐标变换矩阵 RT (将局部坐标系矩阵转换到结构坐标系)
    RT[0:3, 0:3] = R
    RT[3:6, 3:6] = R
    RT[6:9, 6:9] = R
    RT[9:12, 9:12] = R

    L = np.linalg.norm(xm)  # 计算杆件长度
    E = Elastic_modulus
    G = Shear_modulus
    Ix = Inertia_moment[0, 0]
    Iy = Inertia_moment[1, 0]
    Iz = Inertia_moment[2, 0]

    # 定义局部刚度子矩阵(Euler-Bernoulli梁模型, 含瑞丽梁修正)
    # k_ii: 起始节点自身刚度子矩阵
    k_ii = np.array([
        [E*Ax/L,            0,             0,        0,            0,            0],
        [0,      12*E*Iz/L**3,             0,        0,            0,  6*E*Iz/L**2],
        [0,                 0,  12*E*Iy/L**3,        0, -6*E*Iy/L**2,            0],
        [0,                 0,             0,   G*Ix/L,            0,            0],
        [0,                 0, - 6*E*Iy/L**2,        0,     4*E*Iy/L,            0],
        [0,       6*E*Iz/L**2,             0,        0,            0,     4*E*Iz/L]])
    # k_jj: 终止节点自身刚度子矩阵
    k_jj = np.array([
        [E*Ax/L,            0,             0,        0,            0,            0],
        [0,      12*E*Iz/L**3,             0,        0,            0, -6*E*Iz/L**2],
        [0,                 0,  12*E*Iy/L**3,        0,  6*E*Iy/L**2,            0],
        [0,                 0,             0,   G*Ix/L,            0,            0],
        [0,                 0,   6*E*Iy/L**2,        0,     4*E*Iy/L,            0],
        [0,      -6*E*Iz/L**2,             0,        0,            0,      4*E*Iz/L]])
    # k_ij: 起始节点与终止节点之间的耦合刚度子矩阵
    k_ij = np.array([
        [-E*Ax/L,           0,             0,        0,            0,            0],
        [0,     -12*E*Iz/L**3,             0,        0,            0,  6*E*Iz/L**2],
        [0,                 0, -12*E*Iy/L**3,        0, -6*E*Iy/L**2,            0],
        [0,                 0,             0,  -G*Ix/L,            0,            0],
        [0,                 0,   6*E*Iy/L**2,        0,     2*E*Iy/L,            0],
        [0,      -6*E*Iz/L**2,             0,        0,            0,     2*E*Iz/L]])
    # k_ji: 终止节点与起始节点之间的耦合刚度子矩阵
    k_ji = np.array([
        [-E*Ax/L,           0,             0,        0,            0,            0],
        [0,     -12*E*Iz/L**3,             0,        0,            0, -6*E*Iz/L**2],
        [0,                 0, -12*E*Iy/L**3,        0,  6*E*Iy/L**2,            0],
        [0,                 0,             0,  -G*Ix/L,            0,            0],
        [0,                 0,  -6*E*Iy/L**2,        0,     2*E*Iy/L,            0],
        [0,       6*E*Iz/L**2,             0,        0,            0,     2*E*Iz/L]])

    # 考虑预应力刚度(重力引起的非线性项)
    # 取节点坐标中x方向的平均值
    Pre_Stress = -((Length_X - ((pk[0] + pj[0]) / 2)) * Ax * Density + m_Tip) * 9.8

    k0_ii = -Pre_Stress/L * np.array([
        [0,      0,     0,           0,          0,          0],
        [0,    6/5,     0,           0,          0,       L/10],
        [0,      0,   6/5,           0,      -L/10,          0],
        [0,      0,     0,  (Iy+Iz)/Ax,          0,          0],
        [0,      0, -L/10,           0,  2*L**2/15,          0],
        [0,   L/10,     0,           0,          0,  2*L**2/15]])
    k0_jj = -Pre_Stress/L * np.array([
        [0,      0,     0,           0,          0,          0],
        [0,    6/5,     0,           0,          0,      -L/10],
        [0,      0,   6/5,           0,       L/10,          0],
        [0,      0,     0,  (Iy+Iz)/Ax,          0,          0],
        [0,      0,  L/10,           0,  2*L**2/15,          0],
        [0,  -L/10,     0,           0,          0,  2*L**2/15]])
    k0_ij = -Pre_Stress/L * np.array([
        [0,      0,     0,           0,          0,          0],
        [0,   -6/5,     0,           0,          0,       L/10],
        [0,      0,  -6/5,           0,      -L/10,          0],
        [0,      0,     0, -(Iy+Iz)/Ax,          0,          0],
        [0,      0,  L/10,           0,   -L**2/30,          0],
        [0,  -L/10,     0,           0,          0,   -L**2/30]])
    k0_ji = -Pre_Stress/L * np.array([
        [0,      0,     0,           0,          0,          0],
        [0,   -6/5,     0,           0,          0,      -L/10],
        [0,      0,  -6/5,           0,       L/10,          0],
        [0,      0,     0, -(Iy+Iz)/Ax,          0,          0],
        [0,      0, -L/10,           0,   -L**2/30,          0],
        [0,   L/10,     0,           0,          0,   -L**2/30]])

    # 总局部刚度矩阵组装  使用 np.block 进行分块组合
    S_local = np.block([[k_ii + k0_ii, k_ij + k0_ij],
                        [k_ji + k0_ji, k_jj + k0_jj]])
    # 坐标变换至结构(全局)坐标系
    S_Rod = RT.T @ S_local @ RT
    return S_Rod


def space_frame_mass_matrix_rod(NUM_Cell_Rod):
    """
    杆件质量矩阵(全局坐标系下)转换函数
    计算单元坐标系下杆件质量矩阵, 并转换到结构(全局)坐标系下
    参数:
        NUM_Cell_Rod (int): 杆件在单元中的编号 (索引值, 需与 Coordinate_Cell_Rod 数组匹配)
    返回:
        M_Rod (np.ndarray): 12x12 杆件节点的刚度 矩阵, 单位与输入参数保持一致
    """
    # 杆件节点坐标及坐标变换矩阵
    # 获取杆件对应的两个节点编号
    NUM_Node = Coordinate_Cell_Rod[NUM_Cell_Rod, :]  # 应为长度为2的数组
    # 获取杆件两端的节点坐标
    pj = Coordinate_Cell_Node[NUM_Node[0], :]  # 起始节点坐标
    pk = Coordinate_Cell_Node[NUM_Node[1], :]  # 终止节点坐标
    # 计算杆件沿 x 方向的向量及杆件长度 L
    xm = pk - pj
    L = np.linalg.norm(xm)
    # 计算 c = sqrt(xm[0]^2 + xm[2]^2) 用于判断杆件是否接近垂直
    c = np.sqrt(xm[0]**2 + xm[2]**2)
    # 根据 c 的大小确定杆件局部 z 轴方向向量
    if c > 0.0001:  # 若杆件不垂直 则选择一个非平行于 xm 的向量计算局部 z 轴
        zm = np.array([-xm[2], 0, xm[0]])
    else:  # 若杆件接近垂直 则直接取 z 轴单位向量
        zm = np.array([0, 0, 1])
    ym = np.cross(zm, xm)  # 计算局部 y 轴: 取 zm 与 xm 的叉乘
    x_unit = xm / np.linalg.norm(xm)  # 归一化各局部坐标轴
    y_unit = ym / np.linalg.norm(ym)
    z_unit = zm / np.linalg.norm(zm)
    R = np.vstack([x_unit, y_unit, z_unit])  # 形成局部坐标系的旋转矩阵 R (每行为一个单位向量)
    RT = np.zeros((12, 12))  # 构造坐标变换矩阵 RT: 将结构坐标系下 12x12 矩阵划分为4个 3x3 的子块 主对角线上均为 R
    RT[0:3, 0:3] = R
    RT[3:6, 3:6] = R
    RT[6:9, 6:9] = R
    RT[9:12, 9:12] = R

    # 杆件质量矩阵计算
    # 杆件物理参数
    rhoi = Density  # 密度 [kg/m^3]
    Ix = Inertia_moment[0, 0]  # 沿 x 轴的惯性矩 [kg·m^2]
    # Iy = Inertia_moment[1, 0]  # MATLAB中此处 Iy Iz 虽然读取但在后续未使用
    # Iz = Inertia_moment[2, 0]
    Ax_i = Ax  # 与全局截面积 Ax 相同

    # 构造 Euler-Bernoulli 梁的质量矩阵(12x12矩阵) 各项系数按 MATLAB 中给定的数值
    coeff = rhoi * Ax * L / 420.0
    # 乘以系数得到局部质量矩阵
    M_local = coeff * np.array([
        [140,       0,      0,           0,         0,         0,     70,       0,      0,           0,       0,       0],
        [0,       156,      0,           0,         0,      22*L,      0,      54,      0,           0,       0,   -13*L],
        [0,         0,    156,           0,     -22*L,         0,      0,       0,     54,           0,    13*L,       0],
        [0,         0,      0, 140*Ix/Ax_i,         0,         0,      0,       0,      0,  70*Ix/Ax_i,       0,       0],
        [0,         0,  -22*L,           0,    4*L**2,         0,      0,       0,  -13*L,           0, -3*L**2,       0],
        [0,      22*L,      0,           0,         0,    4*L**2,      0,    13*L,      0,           0,       0, -3*L**2],
        [70,        0,      0,           0,         0,         0,    140,       0,      0,           0,       0,       0],
        [0,        54,      0,           0,         0,      13*L,      0,     156,      0,           0,       0,   -22*L],
        [0,         0,     54,           0,     -13*L,         0,      0,       0,    156,           0,    22*L,       0],
        [0,         0,      0,  70*Ix/Ax_i,         0,         0,      0,       0,      0, 140*Ix/Ax_i,       0,       0],
        [0,         0,   13*L,           0,   -3*L**2,         0,      0,       0,   22*L,           0,  4*L**2,       0],
        [0,     -13*L,      0,           0,         0,   -3*L**2,      0,   -22*L,      0,           0,       0,  4*L**2]])
    M_Rod = RT.T @ M_local @ RT  # 将局部坐标系下的质量矩阵转换到结构坐标系下
    return M_Rod


def statics_response_rod(n, nr, crl, K, F):
    """
    静力响应求解函数  结构总节点刚度矩阵方程求解函数
    参数:
        n    : 非约束自由度数
        nr   : 约束自由度数
        crl  : 重排序索引(0-indexed排列, 长度 = n+nr)  由 cumulated_restrained_list 得到
        K    : 全局刚度矩阵  dense格式, 尺寸 (n+nr) x (n+nr)
        F    : 外部载荷向量  dense格式, 长度 n+nr
    返回:
        d : 总节点位移向量(重排后)
        r : 总约束反力向量(重排后)
    """
    total_dofs = n + nr
    d = np.zeros(total_dofs)  # 初始化位移向量 d 和反力向量 r
    r = np.zeros(total_dofs)
    b1 = K[:n, :n]  # 提取非约束部分的刚度矩阵及载荷向量
    b2 = F[:n]
    try:
        d[:n] = np.linalg.solve(b1, b2)
    except np.linalg.LinAlgError:
        d[:n] = np.dot(np.linalg.pinv(b1), b2)
    r[n:total_dofs] = -F[n:total_dofs] + K[n:total_dofs, :n] @ d[:n]  # 计算支座反力
    # 重排总节点位移和反力向量
    d = d[np.array(crl)]
    r = r[np.array(crl)]

    # 还原到原始自由度顺序 注意: crl 是将原始顺序映射为[自由, 约束]的排列向量, 故其逆排列(argsort)将恢复原始顺序
    # inv_crl = np.argsort(np.array(crl))
    # d = d[inv_crl]
    # r = r[inv_crl]

    return d, r


def transient_response_t_newmark(kk, cc, mm, fd, dt, dsp_0, vel_0, acc_0):
    """
    Newmark法计算结构动响应  使用 Newmark 方法计算结构瞬态响应.
    参数:
        kk    : 刚度矩阵
        cc    : 阻尼矩阵
        mm    : 质量矩阵
        fd    : 外载荷向量
        dt    : 时间步长
        dsp_0 : 初始位移向量
        vel_0 : 初始速度向量
        acc_0 : 初始加速度向量
    返回:
        acc_1 : 新时刻加速度
        vel_1 : 新时刻速度
        dsp_1 : 新时刻位移
    """
    alpha = 0.5  # Newmark系数
    beta = 0.5  # Newmark系数
    ekk = kk + mm/(alpha*dt**2) + cc*(beta)/(alpha*dt)  # 有效刚度矩阵: ekk = kk + mm/(alpha*dt^2) + cc*beta/(alpha*dt)
    cfm = dsp_0/(alpha*dt**2) + vel_0/(alpha*dt) + acc_0*(0.5/alpha - 1)  # 计算辅助项 cfm 与 cfc
    cfc = dsp_0*(beta/(alpha*dt)) + vel_0*(beta/alpha - 1) + acc_0*((0.5*beta/alpha - 1)*dt)
    efd = fd + mm @ cfm + cc @ cfc  # 有效载荷向量: efd = fd + mm*cfm + cc*cfc  注意: 此处 mm*cfm 表示矩阵与向量的乘法
    try:  # 求解位移: dsp_1 = ekk\efd
        dsp_1 = np.linalg.solve(ekk, efd)
    except np.linalg.LinAlgError:
        dsp_1 = np.dot(np.linalg.pinv(ekk), efd)
    # 更新加速度: acc_1 = (dsp_1 - dsp_0)/(alpha*dt^2) - vel_0/(alpha*dt) - acc_0*(0.5/alpha-1)
    acc_1 = (dsp_1 - dsp_0)/(alpha*dt**2) - vel_0/(alpha*dt) - acc_0*(0.5/alpha - 1)
    # 更新速度: vel_1 = vel_0 + acc_0*(1-beta)*dt + acc_1*beta*dt
    vel_1 = vel_0 + acc_0*(1-beta)*dt + acc_1*beta*dt
    return acc_1, vel_1, dsp_1


class CantileverBeamModel:
    def __init__(self):
        # 参数加载
        self.disp_tip_z0 = dynamic_conf['Disp_Tip_Z_0']
        self.force_tip_z0 = dynamic_conf['Force_Tip_Z_0']
        self.m_tip = dynamic_conf['m_Tip']
        self.j_tip = dynamic_conf['J_Tip']

        self.Length_X = dynamic_conf['Length_X']
        self.Length_Y = dynamic_conf['Length_Y']
        self.Length_Z = dynamic_conf['Length_Z']
        self.Ax = self.Length_Y * self.Length_Z
        self.Iy = self.Length_Y * self.Length_Z ** 3/12
        self.Iz = self.Length_Y ** 3 * self.Length_Z/12
        self.Ix = self.Iy + self.Iz
        self.Inertia_moment = np.array([[self.Ix], [self.Iy], [self.Iz]])

        self.Elastic_modulus = dynamic_conf['Elastic_modulus']
        self.Density = dynamic_conf['Density']
        self.Poisson_ratio = dynamic_conf['Poisson_ratio']
        self.Shear_modulus = self.Elastic_modulus/(2*(1+self.Poisson_ratio))  # 剪切模量: 由弹性模量和泊松比计算 [Pa]

        # 网格划分
        self.NUM_Cell_Node = dynamic_conf['NUM_Cell_Node']  # 悬臂梁网格划分: 节点数
        self.NUM_Cell_Rod = self.NUM_Cell_Node - 1  # 悬臂梁网格划分: 单元数
        self.n_dof = 6 * self.NUM_Cell_Node  # 每个节点6个自由度

        # 生成几何数据
        self.setup_geometry()
        # 生成约束(此处仅固定根端节点, 即第一个节点6个自由度均固定)
        self.setup_constraints()
        # 组装全局刚度, 质量, 阻尼矩阵
        self.assemble_global_matrices()
        # self.assemble_global_matrices_sparse()

    def setup_geometry(self):
        """ 生成节点坐标及单元连接信息 """
        self.Coordinate_Cell_Node = np.zeros((self.NUM_Cell_Node, 3))  # 生成节点坐标
        # 第一个节点位于原点, 其他节点沿x方向均匀分布
        for i in range(1, self.NUM_Cell_Node):
            self.Coordinate_Cell_Node[i, 0] = self.Length_X / self.NUM_Cell_Rod * i
            # self.Coordinate_Cell_Node[i, 0] = self.Length_X / (self.NUM_Cell_Node - 1) * i
        # 单元连接: 每个单元由相邻两个节点构成
        self.Coordinate_Cell_Rod = np.zeros((self.NUM_Cell_Rod, 2), dtype=int)  # 生成单元连接关系
        for i in range(self.NUM_Cell_Rod):
            # 0-indexed:节点 i 与 i+1 组成单元
            self.Coordinate_Cell_Rod[i, :] = [i, i + 1]
        # 更新全局变量
        global Coordinate_Cell_Node, Coordinate_Cell_Rod
        Coordinate_Cell_Node = self.Coordinate_Cell_Node
        Coordinate_Cell_Rod = self.Coordinate_Cell_Rod

    def setup_constraints(self):
        """构造约束向量, 并重排自由度编号
           此处仅固定根端节点(第0节点)的6个自由度, 其余自由度不约束"""
        self.Node_Constraint = np.zeros(self.n_dof, dtype=int)  # 约束向量表
        # 固定第0节点6个自由度
        self.Node_Constraint[0:6] = 1
        # 重排序, 调用累计约束函数
        N_Free, N_Fix, self.NEW_NodeNum = cumulated_restrained_list(self.n_dof, self.Node_Constraint)
        self.N_Free = N_Free
        self.N_Fix = N_Fix

    def assemble_global_matrices(self):
        """ 组装全局刚度矩阵, 质量矩阵及Rayleigh阻尼矩阵 """
        self.Stiffness_matrix = np.zeros((self.n_dof, self.n_dof))
        self.Mass_matrix = np.zeros((self.n_dof, self.n_dof))
        # 遍历每个杆件单元, 组装刚度和质量矩阵
        for i in range(self.NUM_Cell_Rod):
            # 调用已定义的杆件刚度函数
            Stiffness_Rod = space_frame_stiffness_matrix_rod(i)
            # 对应单元的节点编号(0-indexed)
            rod_nodes = self.Coordinate_Cell_Rod[i, :]
            # 构造该单元对应的自由度索引
            dof_indices = np.hstack([np.arange(6*rod_nodes[0], 6*rod_nodes[0]+6),
                                     np.arange(6*rod_nodes[1], 6*rod_nodes[1]+6)])
            # 组装:累加单元刚度矩阵
            self.Stiffness_matrix[np.ix_(dof_indices, dof_indices)] += Stiffness_Rod

            # 同理, 组装质量矩阵
            Mass_Rod = space_frame_mass_matrix_rod(i)
            self.Mass_matrix[np.ix_(dof_indices, dof_indices)] += Mass_Rod

        # 增加末端附加质量(仅作用于平移自由度)
        Mass_Node_tip = np.diag([self.m_tip, self.m_tip, self.m_tip, 0, 0, 0])
        tip_dof = np.arange(self.n_dof-6, self.n_dof)
        self.Mass_matrix[np.ix_(tip_dof, tip_dof)] += Mass_Node_tip

        # 根据重排序向量 NEW_NodeNum 重排刚度和质量矩阵
        NEW = np.array(self.NEW_NodeNum)
        self.Stiffness_matrix_New = self.Stiffness_matrix[np.ix_(NEW, NEW)]
        self.Mass_matrix_New = self.Mass_matrix[np.ix_(NEW, NEW)]

        # 阻尼矩阵: 采用Rayleigh阻尼模型
        Damping_Coefficients_M = dynamic_conf['Damping_Coefficients_M']
        Damping_Coefficients_K = dynamic_conf['Damping_Coefficients_K']
        self.Damping_matrix = Damping_Coefficients_M * self.Mass_matrix + Damping_Coefficients_K * self.Stiffness_matrix
        self.Damping_matrix_New = self.Damping_matrix[np.ix_(NEW, NEW)]

    def assemble_global_matrices_sparse(self):
        """ 采用稀疏矩阵组装全局刚度与质量矩阵 """
        # 使用列表存储稀疏矩阵的三元组
        stiffness_rows = []
        stiffness_cols = []
        stiffness_data = []

        mass_rows = []
        mass_cols = []
        mass_data = []

        # 对每个单元, 得到局部刚度和质量矩阵(均为12 x 12)
        for elem in range(self.NUM_Cell_Rod):
            # 调用转换后的单元函数(注意: 保证返回的矩阵为np.array)
            Ke = space_frame_stiffness_matrix_rod(elem)
            Me = space_frame_mass_matrix_rod(elem)
            # 单元节点编号(0-indexed)
            nodes = self.Coordinate_Cell_Rod[elem, :]
            dof_indices = np.hstack([np.arange(6*nodes[0], 6*nodes[0]+6),
                                     np.arange(6*nodes[1], 6*nodes[1]+6)])
            # 将Ke,Me的每个非零项存入列表
            for i_local, I in enumerate(dof_indices):
                for j_local, J in enumerate(dof_indices):
                    stiffness_rows.append(I)
                    stiffness_cols.append(J)
                    stiffness_data.append(Ke[i_local, j_local])

                    mass_rows.append(I)
                    mass_cols.append(J)
                    mass_data.append(Me[i_local, j_local])

        # 末端附加质量(仅平移分量)
        tip_dof = np.arange(self.n_dof-6, self.n_dof)
        Mass_tip = np.diag([self.m_tip, self.m_tip, self.m_tip, 0, 0, 0])
        for i_local, I in enumerate(tip_dof):
            for j_local, J in enumerate(tip_dof):
                stiffness_rows.append(I)
                stiffness_cols.append(J)
                stiffness_data.append(0)  # 刚度增加为0
                mass_rows.append(I)
                mass_cols.append(J)
                mass_data.append(Mass_tip[i_local, j_local])

        # 构造稀疏矩阵
        self.Stiffness_matrix = csr_matrix((stiffness_data, (stiffness_rows, stiffness_cols)),
                                           shape=(self.n_dof, self.n_dof))
        self.Mass_matrix = csr_matrix((mass_data, (mass_rows, mass_cols)),
                                      shape=(self.n_dof, self.n_dof))

        # 重排矩阵: NEW_NodeNum为列表, 构造置换矩阵 P(dense版也可以, 用于稀疏矩阵重排)
        NEW = np.array(self.NEW_NodeNum)
        self.Stiffness_matrix_New = self.Stiffness_matrix[NEW, :][:, NEW]
        self.Mass_matrix_New = self.Mass_matrix[NEW, :][:, NEW]

        # 阻尼矩阵: Rayleigh阻尼
        Damping_Coefficients_M = dynamic_conf['Damping_Coefficients_M']
        Damping_Coefficients_K = dynamic_conf['Damping_Coefficients_K']
        self.Damping_matrix = Damping_Coefficients_M * self.Mass_matrix + Damping_Coefficients_K * self.Stiffness_matrix
        self.Damping_matrix_New = self.Damping_matrix[NEW, :][:, NEW]

    def modal_analysis(self):
        """ 求解自由自由度部分的固有频率, 并输出前10个频率(Hz) """
        # 只考虑自由自由度(非约束)
        K_free = self.Stiffness_matrix_New[:self.N_Free, :self.N_Free]
        M_free = self.Mass_matrix_New[:self.N_Free, :self.N_Free]
        # 求解广义特征值问题
        eigvals, _ = eig(K_free, M_free)
        # 取实部正值
        w = np.sqrt(np.abs(eigvals)) / (2*np.pi)
        freqs = np.sort(w.real)
        print(f"前10个固有频率 (Hz):\n{freqs[:10]}")
        return freqs

    # def modal_analysis(self, num_modes=10):
    #     """
    #     利用稀疏特征值求解器求解前 num_modes 个固有频率
    #     这里通过对自由度刚度矩阵添加微小正则化项避免完全奇异的情况,
    #     并采用 'SM'(Smallest Magnitude)模式求解最小模特征值
    #     """
    #     # 提取自由度部分刚度和质量矩阵, 转换为 CSC 格式
    #     K_free = self.Stiffness_matrix_New[:self.N_Free, :self.N_Free].tocsc()
    #     M_free = self.Mass_matrix_New[:self.N_Free, :self.N_Free].tocsc()
    #     # 添加一个极小的正则化项, 避免刚度矩阵奇异
    #     from scipy.sparse import identity
    #     reg = 1e-6 * identity(K_free.shape[0], format='csc')
    #     K_reg = K_free + reg
    #     # 采用 eigsh 求解最小的 num_modes 个特征值
    #     try:
    #         eigvals, _ = eigsh(K_reg, k=num_modes, M=M_free, which='SM')
    #     except Exception as e:
    #         print("eigsh 求解出错:", e)
    #         return None
    #     # 计算频率, 单位 Hz
    #     freqs = np.sort(np.sqrt(np.abs(eigvals.real)))/(2*np.pi)
    #     print("前%d个固有频率 (Hz):" % num_modes)
    #     print(freqs)
    #     return freqs

    def static_analysis(self):
        """ 基于静力学求解初始位移响应(当施加位移载荷时, 转化为等效力) """
        # 如果仅施加位移载荷, 计算等效力
        if self.force_tip_z0 == 0 and self.disp_tip_z0 != 0:
            # 使用公式: Force = 3*δ*Iy*E / L^3
            # 注意: Iy存储在 Inertia_moment 的第二行(0-indexed行1)
            self.force_tip_z0 = 3 * self.disp_tip_z0 * self.Inertia_moment[1, 0] * self.Elastic_modulus / (self.Length_X**3)
        # 构造全局载荷向量: 仅末端节点在z方向施加力
        Load_0 = np.zeros((self.n_dof, 1))
        # 末端节点的z方向自由度: 对于最后一个节点, 其z自由度索引为 6*(NUM_Cell_Node-1)+2
        tip_z_index = 6*(self.NUM_Cell_Node-1) + 2
        Load_0[tip_z_index, 0] = self.force_tip_z0
        # 重排载荷
        NEW = np.array(self.NEW_NodeNum)
        Load_0_new = Load_0.copy()
        Load_0_new[NEW, 0] = Load_0[:, 0]
        # 调用静力响应求解(此处用稀疏矩阵可以先转为dense求解)
        # K_dense = self.Stiffness_matrix_New.toarray()
        # F_dense = Load_0_new[:, 0]
        # 调用静力学响应函数
        disp_static, _ = statics_response_rod(self.N_Free, self.N_Fix, NEW, self.Stiffness_matrix_New, Load_0_new[:, 0])
        # disp_static, _ = statics_response_rod(self.N_Free, self.N_Fix, NEW, K_dense, F_dense)
        self.Displacement_static = disp_static
        return disp_static

    def dynamic_response(self):
        """利用Newmark法求解结构动力响应
           total_time: 总时长 [s]
           num_steps : 时间步数
        """
        total_time = dynamic_conf['Total_time']
        num_steps = dynamic_conf['Num_Steps']
        dt = total_time/(num_steps-1)
        Time_span = np.linspace(0, total_time, num_steps)  # 时间离散

        # 初始化响应记录矩阵
        Displacement = np.zeros((num_steps, self.n_dof))
        Velocity = np.zeros((num_steps, self.n_dof))
        Acceleration = np.zeros((num_steps, self.n_dof))
        Force_Node = np.zeros((self.n_dof, num_steps))

        # 重力项(设为零 如需考虑可修改)
        # Acc_gravity = np.zeros((self.n_dof,))
        # 重力考虑(此处暂设为零, 也可修改)
        Acceleration_gravity = np.zeros((self.n_dof, 1))
        Force_Node_gravity = self.Mass_matrix @ Acceleration_gravity

        # 外载荷转换: 若位移载荷存在, 则已转换为等效力
        Load_0 = np.zeros((self.n_dof, 1))
        tip_z_index = 6*(self.NUM_Cell_Node-1) + 2
        Load_0[tip_z_index, 0] = self.force_tip_z0
        NEW = np.array(self.NEW_NodeNum)
        Load_0_new = Load_0.copy()
        Load_0_new[NEW, 0] = Load_0[:, 0]

        # 静力响应求解作为初始条件
        dsp_0, _ = statics_response_rod(self.N_Free, self.N_Fix, NEW, self.Stiffness_matrix_New, Load_0_new[:, 0])
        # dsp_0, _ = statics_response_rod(self.N_Free, self.N_Fix, NEW, self.Stiffness_matrix_New[:self.N_Free, :self.N_Free].toarray(), Load_0_new[:self.N_Free, 0])
        Displacement[0, :] = dsp_0
        # 初始速度与加速度设为0
        vel_0 = np.zeros((self.N_Free,))
        acc_0 = np.zeros((self.N_Free,))

        # 主时间积分循环(仅对自由自由度部分计算, 约束自由度置零)
        for jj in range(1, num_steps):
            # 提取上一时刻的响应(重排后自由度顺序)
            dsp_prev = Displacement[jj-1, :].reshape(-1, 1)
            # 重排到原排序
            dsp_0_free = dsp_prev[NEW][:self.N_Free].flatten()
            # 同理速度与加速度
            vel_prev = Velocity[jj-1, :].reshape(-1, 1)
            vel_0_free = vel_prev[NEW][:self.N_Free].flatten()
            acc_prev = Acceleration[jj-1, :].reshape(-1, 1)
            acc_0_free = acc_prev[NEW][:self.N_Free].flatten()  # 此处不加重力(已在载荷中考虑)

            # 外载荷更新: 此处假设外载荷恒定(可扩展为时变载荷)
            Force_New = np.zeros((self.n_dof, 1))
            Force_New[:, 0] = Force_Node[:, jj]
            Force_New_new = Force_New.copy()
            Force_New_new[NEW, 0] = Force_New[:, 0]
            # Force_free = Force_New_new[:self.N_Free, 0] + Force_Node_gravity = Force_Node_gravity[:self.N_Free, 0] if Force_Node_gravity.shape[0] >= self.N_Free else 0
            # 注意:上行代码兼容性处理, 若重力项为空则为0
            if Force_Node_gravity.shape[0] >= self.N_Free:
                Force_free = Force_New_new[:self.N_Free, 0] + Force_Node_gravity[:self.N_Free, 0]
            else:
                Force_free = Force_New_new[:self.N_Free, 0]

            # F_free = Load_0_new[:self.N_Free, 0]  # 此处不含重力项
            # 只针对自由自由度部分, 调用Newmark法
            K_free = self.Stiffness_matrix_New[:self.N_Free, :self.N_Free]
            C_free = self.Damping_matrix_New[:self.N_Free, :self.N_Free]
            M_free = self.Mass_matrix_New[:self.N_Free, :self.N_Free]
            acc_new, vel_new, dsp_new = transient_response_t_newmark(K_free, C_free, M_free,
                                                                     Force_free, dt,
                                                                     dsp_0_free, vel_0_free, acc_0_free)
            # # 调用Newmark求解(内部已采用 np.linalg.solve, 如必要可替换为 spsolve)
            # acc_new, vel_new, dsp_new = transient_response_t_newmark(K_free.toarray(), C_free.toarray(), M_free.toarray(),
            #                                                          F_free, dt, dsp_0[:self.N_Free], vel_0, acc_0)

            # # 更新初始条件
            # dsp_0[:self.N_Free] = dsp_new
            # vel_0 = vel_new
            # acc_0 = acc_new
            # 将计算结果置入自由度部分, 其余(约束)置零
            dsp_full = np.zeros((self.N_Free+self.N_Fix,))
            dsp_full[:self.N_Free] = dsp_new
            vel_full = np.zeros((self.N_Free+self.N_Fix,))
            vel_full[:self.N_Free] = vel_new
            acc_full = np.zeros((self.N_Free+self.N_Fix,))
            acc_full[:self.N_Free] = acc_new

            # 重排回全局顺序
            dsp_global = np.zeros((self.n_dof,))
            vel_global = np.zeros((self.n_dof,))
            acc_global = np.zeros((self.n_dof,))
            dsp_global[np.array(NEW)] = dsp_full
            vel_global[np.array(NEW)] = vel_full
            acc_global[np.array(NEW)] = acc_full

            Displacement[jj, :] = dsp_global
            Velocity[jj, :] = vel_global
            Acceleration[jj, :] = acc_global
            print(f"Time step {jj}/{num_steps}")

        self.Time_span = Time_span
        self.Displacement = Displacement
        self.Velocity = Velocity
        self.Acceleration = Acceleration
        return Time_span, Displacement

    def output_results(self, filename='dynamic/perturbation_tip.txt'):
        """ 输出末端位移(单位:mm)到文本文件, 并绘制曲线图 """
        # 末端节点: 对于最后一个节点, 其z方向自由度索引为 6*(NUM_Cell_Node-1)+2
        tip_z_index = 6*(self.NUM_Cell_Node-1) + 2
        # 注意: 由于全局重排, 实际Tip位移需根据NEW_NodeNum重排
        tip_disp = self.Displacement[:, tip_z_index] * 1000  # 末端的位移响应 转换为mm

        np.savetxt(filename, tip_disp, fmt='%.4f')
        print(f"Tip perturbation saved to {filename}")

        plt.figure()
        plt.plot(self.Time_span, tip_disp, 'b-', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Tip displacement [mm]')
        plt.title('Dynamic Response at Cantilever Tip')
        plt.show()

    def run(self):
        """ 依次执行模态分析, 静力学求解, 动态响应求解及结果输出 """
        self.modal_analysis()
        self.static_analysis()
        self.dynamic_response()
        self.output_results()


if __name__ == '__main__':

    model = CantileverBeamModel()
    model.run()
