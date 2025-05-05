clear
% close all
% clc

global NUM_Cell_Rod NUM_Cell_Node Coordinate_Cell_Node Coordinate_Cell_Rod
global Length_X Length_Y Length_Z Ax m_Tip J_Tip
global Elastic_modulus  Shear_modulus  Inertia_moment Density poisson_ratio 

%% 载荷输入
Disp_Tip_Z_0 = 45*10^-3;   % 悬臂梁末端施加的初始位移
Force_Tip_Z_0 = 0;         % 悬臂梁末端施加的初始力
m_Tip = 0.0;               % 末端质量块的重量
J_Tip = 0;                 % 末端质量块的转动惯量
%% 几何和参数的定义
Length_X = 485*10^-3;          % 悬臂梁长 X 方向【m】
Length_Y = 10*10^-3;           % 悬臂梁宽 Y 方向【m】
Length_Z = 2*10^-3;            % 悬臂梁厚 Z 方向【m】

Elastic_modulus = 71e9;        % 弹性模量E【Pa】
Density =2800 ;                % 材料密度ρ【kg/m^3】
poisson_ratio = 0.33;          % 泊松比
Shear_modulus = Elastic_modulus/(2*(1+poisson_ratio));   % 剪切模量：由弹性模量和泊松比计算

Ax = Length_Z*Length_Y;                           % 横截面积：矩形
Inertia_moment = zeros(3,1);                      % 杆件各个方向的惯性矩
Inertia_moment(2,1) = Length_Y*Length_Z^3/12;     % 沿y轴的惯性矩
Inertia_moment(3,1) = Length_Y^3*Length_Z/12;     % 沿z轴的惯性矩
Inertia_moment(1,1) = Inertia_moment(2,1) + Inertia_moment(3,1);     % 沿x轴的惯性矩: Ip=Ix+Iy

%% 单元离散，网格划分，坐标求解
NUM_Cell_Node  = 201;                   % 悬臂梁网格划分：节点数
NUM_Cell_Rod   = NUM_Cell_Node-1;       % 悬臂梁网格划分：单元数    
Coordinate_Cell_Node = zeros(NUM_Cell_Node,3);  % 悬臂梁网格划分：节点坐标
for i=2:NUM_Cell_Node
    Coordinate_Cell_Node(i,1) = Length_X/NUM_Cell_Rod*(i-1);
end

Coordinate_Cell_Rod = zeros(NUM_Cell_Rod,2);    % 悬臂梁网格划分：单元节点编号
for i=1:NUM_Cell_Node
    Coordinate_Cell_Rod(i,:) = [i i+1;];
end

%% 约束向量表
Node_Constraint(1:6*NUM_Cell_Node) = 0;   % 约束向量表
NUM_Node_Constraint = 1;                  % 约束节点数目
for i=1:NUM_Node_Constraint
    k_Fix = 1;                            % 受约束节点编号
    Node_Constraint(6*k_Fix-5:6*k_Fix) = [1 1 1 1 1 1;];  % 该节点6个自由度约束状况
end
[N_Free,N_Fix,NEW_NodeNum] = CumulatedRestrainedList(6*NUM_Cell_Node,Node_Constraint);   % 根据约束情况对结点进行重新排序编号

%% 刚度矩阵计算
Stiffness_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node) = 0;         % 【全局坐标系】下的【整体刚度矩阵】：总结点数6倍的方阵
for i=1:NUM_Cell_Rod
    Stiffness_Rod = SpaceFrame_StiffnessMatrix_Rod(i);         % 杆件在【全局坐标系】下的刚度矩阵[12×12]
    Rod_Node = Coordinate_Cell_Rod(i,:);                      % 杆件节点编号
    j=[6*Rod_Node(1)-5:6*Rod_Node(1) 6*Rod_Node(2)-5:6*Rod_Node(2)];           % 节点编号×6，对应每个节点的6个自由度
    Stiffness_matrix(j(1:12),j(1:12)) = Stiffness_matrix(j(1:12),j(1:12)) + Stiffness_Rod(1:12,1:12);   % 整体结构的刚度矩阵组装
end
Stiffness_matrix_New(NEW_NodeNum(1:6*NUM_Cell_Node),NEW_NodeNum(1:6*NUM_Cell_Node)) = Stiffness_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node);

%% 质量矩阵计算
Mass_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node) = 0;      % 【全局坐标系】下的【整体质量矩阵】：总结点数6倍的方阵
for i=1:NUM_Cell_Rod
    Mass_Rod = SpaceFrame_MassMatrix_Rod(i);               % 【全局坐标系】下的【杆件质量矩阵】[12×12]
    Rod_Node = Coordinate_Cell_Rod(i,:);                   % 杆件节点编号
    j=[6*Rod_Node(1)-5:6*Rod_Node(1) 6*Rod_Node(2)-5:6*Rod_Node(2)];          % 节点编号×6，对应每个节点的6个自由度
    Mass_matrix(j(1:12),j(1:12)) = Mass_matrix(j(1:12),j(1:12)) + Mass_Rod(1:12,1:12);     % 整体结构的质量矩阵组装
end

Mass_Node_tip = diag([m_Tip m_Tip m_Tip 0 0 0]);

Mass_matrix(6*NUM_Cell_Node-5:6*NUM_Cell_Node,6*NUM_Cell_Node-5:6*NUM_Cell_Node) = Mass_matrix(6*NUM_Cell_Node-5:6*NUM_Cell_Node,6*NUM_Cell_Node-5:6*NUM_Cell_Node) + Mass_Node_tip(1:6,1:6);
Mass_matrix_New(NEW_NodeNum(1:6*NUM_Cell_Node),NEW_NodeNum(1:6*NUM_Cell_Node)) = Mass_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node);

%% 阻尼矩阵计算（采用比例阻尼模型——Rayleigh模型）
Damping_Coefficients_M = 0.0001;   %0.02593;
Damping_Coefficients_K = 0.00025;   %0.01390;
Damping_matrix = Damping_Coefficients_M*Mass_matrix + Damping_Coefficients_K*Stiffness_matrix;
Damping_matrix_New(NEW_NodeNum(1:6*NUM_Cell_Node),NEW_NodeNum(1:6*NUM_Cell_Node)) = Damping_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node);

%% 频率求解
[Eigen_Vector,Eigen_Value]=eig(Stiffness_matrix_New(1:N_Free,1:N_Free),Mass_matrix_New(1:N_Free,1:N_Free));
w=realsqrt(abs(Eigen_Value))/2/pi;
[Frequence,Index]=sort(diag(w));
Frequence(1:10)

%% 动力学响应求解
NUM_Step = 1001;                        % 总的时间步数
Time = 10;                             % 总的时长
Time_span = linspace(0,Time,NUM_Step);  % 时间离散

% 初始化
Displacement = zeros(NUM_Step,6*NUM_Cell_Node);  % 节点位移存储
Velocity     = zeros(NUM_Step,6*NUM_Cell_Node);  % 节点速度存储
Acceleration = zeros(NUM_Step,6*NUM_Cell_Node);  % 节点加速度存储
Force_Node   = zeros(6*NUM_Cell_Node,NUM_Step);  % 节点外载荷存储

%% 重力的考量
Acceleration_gravity = zeros(6*NUM_Cell_Node,1);  % 重力加速度
% for ii=1:NUM_Cell_Node
%     Acceleration_gravity(6*(ii-1)+1,1)=9.8;
% end
% Force_Node_gravity = zeros(6*NUM_Cell_Node,1);    % 重力
Force_Node_gravity=Mass_matrix*Acceleration_gravity;
%% 外载荷——位移载荷转换
if Force_Tip_Z_0==0 && Disp_Tip_Z_0~=0
    Force_Tip_Z_0 = 3*Disp_Tip_Z_0*Inertia_moment(2,1)*Elastic_modulus/Length_X^3;
end

Load_0 = zeros(6*NUM_Cell_Node,1);          % 施加的力载荷
Load_0(6*NUM_Cell_Node-3,1)= Force_Tip_Z_0;
Load_0(NEW_NodeNum(1:6*NUM_Cell_Node),1) = Load_0(1:6*NUM_Cell_Node,1);
[Displacement_0, ~]= Statics_Response_Rod(N_Free,N_Fix,NEW_NodeNum,Stiffness_matrix_New,Load_0(:,1));   % 求解给定力载荷后的位移
Displacement(1,:)=Displacement_0';

%% Newmark法求解动力学响应
for jj=2:NUM_Step
    dsp_0(NEW_NodeNum(1:6*NUM_Cell_Node),1) = Displacement(jj-1,:)';
    vel_0(NEW_NodeNum(1:6*NUM_Cell_Node),1) = Velocity    (jj-1,:)';
    acc_0(NEW_NodeNum(1:6*NUM_Cell_Node),1) = Acceleration(jj-1,:)' + Acceleration_gravity(:,:);
    Force_New(NEW_NodeNum(1:6*NUM_Cell_Node),1) = Force_Node(1:6*NUM_Cell_Node,jj) + Force_Node_gravity(:,:);

    [acc,vel,dsp] = Transient_ResponseT_Newmark(Stiffness_matrix_New(1:N_Free,1:N_Free),Damping_matrix_New(1:N_Free,1:N_Free),Mass_matrix_New(1:N_Free,1:N_Free),...
                                                Force_New(1:N_Free),Time_span(2),dsp_0(1:N_Free),vel_0(1:N_Free),acc_0(1:N_Free));
    dsp(N_Free+1:N_Fix+N_Free) = 0;
    vel(N_Free+1:N_Fix+N_Free) = 0;
    acc(N_Free+1:N_Fix+N_Free) = 0;
    
    Displacement(jj,:) = dsp(NEW_NodeNum(1:6*NUM_Cell_Node))';
    Velocity    (jj,:) = vel(NEW_NodeNum(1:6*NUM_Cell_Node))';
    Acceleration(jj,:) = acc(NEW_NodeNum(1:6*NUM_Cell_Node))';
    jj
end
%% 结果输出
Perturbation_Tip_Z = Displacement(:,6*NUM_Cell_Node-3)*1000;   % 末端的位移响应【mm】

figure
plot(Time_span,Perturbation_Tip_Z)


