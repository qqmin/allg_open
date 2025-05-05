clear
% close all
% clc

global NUM_Cell_Rod NUM_Cell_Node Coordinate_Cell_Node Coordinate_Cell_Rod
global Length_X Length_Y Length_Z Ax m_Tip J_Tip
global Elastic_modulus  Shear_modulus  Inertia_moment Density poisson_ratio 

%% �غ�����
Disp_Tip_Z_0 = 45*10^-3;   % ������ĩ��ʩ�ӵĳ�ʼλ��
Force_Tip_Z_0 = 0;         % ������ĩ��ʩ�ӵĳ�ʼ��
m_Tip = 0.0;               % ĩ�������������
J_Tip = 0;                 % ĩ���������ת������
%% ���κͲ����Ķ���
Length_X = 485*10^-3;          % �������� X ����m��
Length_Y = 10*10^-3;           % �������� Y ����m��
Length_Z = 2*10^-3;            % �������� Z ����m��

Elastic_modulus = 71e9;        % ����ģ��E��Pa��
Density =2800 ;                % �����ܶȦѡ�kg/m^3��
poisson_ratio = 0.33;          % ���ɱ�
Shear_modulus = Elastic_modulus/(2*(1+poisson_ratio));   % ����ģ�����ɵ���ģ���Ͳ��ɱȼ���

Ax = Length_Z*Length_Y;                           % ������������
Inertia_moment = zeros(3,1);                      % �˼���������Ĺ��Ծ�
Inertia_moment(2,1) = Length_Y*Length_Z^3/12;     % ��y��Ĺ��Ծ�
Inertia_moment(3,1) = Length_Y^3*Length_Z/12;     % ��z��Ĺ��Ծ�
Inertia_moment(1,1) = Inertia_moment(2,1) + Inertia_moment(3,1);     % ��x��Ĺ��Ծ�: Ip=Ix+Iy

%% ��Ԫ��ɢ�����񻮷֣��������
NUM_Cell_Node  = 201;                   % ���������񻮷֣��ڵ���
NUM_Cell_Rod   = NUM_Cell_Node-1;       % ���������񻮷֣���Ԫ��    
Coordinate_Cell_Node = zeros(NUM_Cell_Node,3);  % ���������񻮷֣��ڵ�����
for i=2:NUM_Cell_Node
    Coordinate_Cell_Node(i,1) = Length_X/NUM_Cell_Rod*(i-1);
end

Coordinate_Cell_Rod = zeros(NUM_Cell_Rod,2);    % ���������񻮷֣���Ԫ�ڵ���
for i=1:NUM_Cell_Node
    Coordinate_Cell_Rod(i,:) = [i i+1;];
end

%% Լ��������
Node_Constraint(1:6*NUM_Cell_Node) = 0;   % Լ��������
NUM_Node_Constraint = 1;                  % Լ���ڵ���Ŀ
for i=1:NUM_Node_Constraint
    k_Fix = 1;                            % ��Լ���ڵ���
    Node_Constraint(6*k_Fix-5:6*k_Fix) = [1 1 1 1 1 1;];  % �ýڵ�6�����ɶ�Լ��״��
end
[N_Free,N_Fix,NEW_NodeNum] = CumulatedRestrainedList(6*NUM_Cell_Node,Node_Constraint);   % ����Լ������Խ���������������

%% �նȾ������
Stiffness_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node) = 0;         % ��ȫ������ϵ���µġ�����նȾ��󡿣��ܽ����6���ķ���
for i=1:NUM_Cell_Rod
    Stiffness_Rod = SpaceFrame_StiffnessMatrix_Rod(i);         % �˼��ڡ�ȫ������ϵ���µĸնȾ���[12��12]
    Rod_Node = Coordinate_Cell_Rod(i,:);                      % �˼��ڵ���
    j=[6*Rod_Node(1)-5:6*Rod_Node(1) 6*Rod_Node(2)-5:6*Rod_Node(2)];           % �ڵ��š�6����Ӧÿ���ڵ��6�����ɶ�
    Stiffness_matrix(j(1:12),j(1:12)) = Stiffness_matrix(j(1:12),j(1:12)) + Stiffness_Rod(1:12,1:12);   % ����ṹ�ĸնȾ�����װ
end
Stiffness_matrix_New(NEW_NodeNum(1:6*NUM_Cell_Node),NEW_NodeNum(1:6*NUM_Cell_Node)) = Stiffness_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node);

%% �����������
Mass_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node) = 0;      % ��ȫ������ϵ���µġ������������󡿣��ܽ����6���ķ���
for i=1:NUM_Cell_Rod
    Mass_Rod = SpaceFrame_MassMatrix_Rod(i);               % ��ȫ������ϵ���µġ��˼���������[12��12]
    Rod_Node = Coordinate_Cell_Rod(i,:);                   % �˼��ڵ���
    j=[6*Rod_Node(1)-5:6*Rod_Node(1) 6*Rod_Node(2)-5:6*Rod_Node(2)];          % �ڵ��š�6����Ӧÿ���ڵ��6�����ɶ�
    Mass_matrix(j(1:12),j(1:12)) = Mass_matrix(j(1:12),j(1:12)) + Mass_Rod(1:12,1:12);     % ����ṹ������������װ
end

Mass_Node_tip = diag([m_Tip m_Tip m_Tip 0 0 0]);

Mass_matrix(6*NUM_Cell_Node-5:6*NUM_Cell_Node,6*NUM_Cell_Node-5:6*NUM_Cell_Node) = Mass_matrix(6*NUM_Cell_Node-5:6*NUM_Cell_Node,6*NUM_Cell_Node-5:6*NUM_Cell_Node) + Mass_Node_tip(1:6,1:6);
Mass_matrix_New(NEW_NodeNum(1:6*NUM_Cell_Node),NEW_NodeNum(1:6*NUM_Cell_Node)) = Mass_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node);

%% ���������㣨���ñ�������ģ�͡���Rayleighģ�ͣ�
Damping_Coefficients_M = 0.0001;   %0.02593;
Damping_Coefficients_K = 0.00025;   %0.01390;
Damping_matrix = Damping_Coefficients_M*Mass_matrix + Damping_Coefficients_K*Stiffness_matrix;
Damping_matrix_New(NEW_NodeNum(1:6*NUM_Cell_Node),NEW_NodeNum(1:6*NUM_Cell_Node)) = Damping_matrix(1:6*NUM_Cell_Node,1:6*NUM_Cell_Node);

%% Ƶ�����
[Eigen_Vector,Eigen_Value]=eig(Stiffness_matrix_New(1:N_Free,1:N_Free),Mass_matrix_New(1:N_Free,1:N_Free));
w=realsqrt(abs(Eigen_Value))/2/pi;
[Frequence,Index]=sort(diag(w));
Frequence(1:10)

%% ����ѧ��Ӧ���
NUM_Step = 1001;                        % �ܵ�ʱ�䲽��
Time = 10;                             % �ܵ�ʱ��
Time_span = linspace(0,Time,NUM_Step);  % ʱ����ɢ

% ��ʼ��
Displacement = zeros(NUM_Step,6*NUM_Cell_Node);  % �ڵ�λ�ƴ洢
Velocity     = zeros(NUM_Step,6*NUM_Cell_Node);  % �ڵ��ٶȴ洢
Acceleration = zeros(NUM_Step,6*NUM_Cell_Node);  % �ڵ���ٶȴ洢
Force_Node   = zeros(6*NUM_Cell_Node,NUM_Step);  % �ڵ����غɴ洢

%% �����Ŀ���
Acceleration_gravity = zeros(6*NUM_Cell_Node,1);  % �������ٶ�
% for ii=1:NUM_Cell_Node
%     Acceleration_gravity(6*(ii-1)+1,1)=9.8;
% end
% Force_Node_gravity = zeros(6*NUM_Cell_Node,1);    % ����
Force_Node_gravity=Mass_matrix*Acceleration_gravity;
%% ���غɡ���λ���غ�ת��
if Force_Tip_Z_0==0 && Disp_Tip_Z_0~=0
    Force_Tip_Z_0 = 3*Disp_Tip_Z_0*Inertia_moment(2,1)*Elastic_modulus/Length_X^3;
end

Load_0 = zeros(6*NUM_Cell_Node,1);          % ʩ�ӵ����غ�
Load_0(6*NUM_Cell_Node-3,1)= Force_Tip_Z_0;
Load_0(NEW_NodeNum(1:6*NUM_Cell_Node),1) = Load_0(1:6*NUM_Cell_Node,1);
[Displacement_0, ~]= Statics_Response_Rod(N_Free,N_Fix,NEW_NodeNum,Stiffness_matrix_New,Load_0(:,1));   % ���������غɺ��λ��
Displacement(1,:)=Displacement_0';

%% Newmark����⶯��ѧ��Ӧ
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
%% ������
Perturbation_Tip_Z = Displacement(:,6*NUM_Cell_Node-3)*1000;   % ĩ�˵�λ����Ӧ��mm��

figure
plot(Time_span,Perturbation_Tip_Z)


