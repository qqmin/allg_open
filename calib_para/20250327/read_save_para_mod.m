%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 读取calibrationSession.mat中指定标定参数并保存至文本calib_para.txt中
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;
% 设置显示格式为长格式format函数来设置输出格式 以提高小数点的精度
% https://bbs.pinggu.org/thread-3002949-1-1.html
format long g;

calib_data = load('calibrationSession.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 写入文件路径
% https://www.cnblogs.com/tsingke/p/13344020.html
calib_para_doc = fopen('calib_para.txt', 'wt');


ImageSize = calib_data.calibrationSession.CameraParameters.CameraParameters1.ImageSize;
fprintf('ImageSize_Height: %d, ImageSize_Width: %d', ImageSize(1,1), ImageSize(1,2));
fprintf(calib_para_doc, '%s\n', 'ImageSize(HxW):');
fprintf(calib_para_doc, '%d\t', ImageSize(1,1));
fprintf(calib_para_doc, '%d', ImageSize(1,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 显示并保存left参数 K & Distortion
K_left = calib_data.calibrationSession.CameraParameters.CameraParameters1.K;
RadialDist_left = calib_data.calibrationSession.CameraParameters.CameraParameters1.RadialDistortion;
TangenDist_left = calib_data.calibrationSession.CameraParameters.CameraParameters1.TangentialDistortion;
disp(' ')
disp('K_left:')
disp(K_left)

Dist_left(1,1) = RadialDist_left(1,1);
Dist_left(1,2) = RadialDist_left(1,2);
Dist_left(1,3) = TangenDist_left(1,1);
Dist_left(1,4) = TangenDist_left(1,2);
Dist_left(1,5) = 0.0;
disp('Dist_left:')
disp(Dist_left)

% 保存left参数 K & Distortion
[K_left_m, K_left_n] = size(K_left);
% fprintf('K_left_m: %d, K_left_n: %d\n\n', K_left_m, K_left_n);

fprintf(calib_para_doc, '\n%s\n', 'K_left:');
for i = 1:1:K_left_m
    for j = 1:1:K_left_n
        if j == K_left_n
            fprintf(calib_para_doc, '%15.15f\n', K_left(i,j)); % 输出格式为浮点数,占用15位,保留15位小数
        else
            fprintf(calib_para_doc, '%15.15f\t', K_left(i,j));
        end
    end
end

fprintf(calib_para_doc, '%s\n', 'Dist_left:');
for i = 1:1:4
    fprintf(calib_para_doc, '%15.15f\t', Dist_left(1,i));
end
fprintf(calib_para_doc, '%15.15f', Dist_left(1,5));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 显示并保存right参数 K & Distortion
K_right = calib_data.calibrationSession.CameraParameters.CameraParameters2.K;
RadialDist_right = calib_data.calibrationSession.CameraParameters.CameraParameters2.RadialDistortion;
TangenDist_right = calib_data.calibrationSession.CameraParameters.CameraParameters2.TangentialDistortion;
disp('K_right:')
disp(K_right)

Dist_right(1,1) = RadialDist_right(1,1);
Dist_right(1,2) = RadialDist_right(1,2);
Dist_right(1,3) = TangenDist_right(1,1);
Dist_right(1,4) = TangenDist_right(1,2);
Dist_right(1,5) = 0.0;
disp('Dist_right:')
disp(Dist_right)

% 保存right参数 K & Distortion
[K_right_m, K_right_n] = size(K_right);
% fprintf('K_right_m: %d, K_right_n: %d\n\n', K_right_m, K_right_n);

fprintf(calib_para_doc, '\n%s\n', 'K_right:');
for i = 1:1:K_right_m
    for j = 1:1:K_right_n
        if j == K_right_n
            fprintf(calib_para_doc, '%15.15f\n', K_right(i,j)); % 输出格式为浮点数,占用15位,保留15位小数
        else
            fprintf(calib_para_doc, '%15.15f\t', K_right(i,j));
        end
    end
end

fprintf(calib_para_doc, '%s\n', 'Dist_right:');
for i = 1:1:4
    fprintf(calib_para_doc, '%15.15f\t', Dist_right(1,i));
end
fprintf(calib_para_doc, '%15.15f', Dist_right(1,5));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 显示并保存stereo参数 PoseCamera2_A
PoseCamera2_A = calib_data.calibrationSession.CameraParameters.PoseCamera2.A;
disp('PoseCamera2_A:')
disp(PoseCamera2_A)

[PoseCamera2_A_m, PoseCamera2_A_n] = size(PoseCamera2_A);
% fprintf('PoseCamera2_A_m: %d, PoseCamera2_A_n: %d\n', PoseCamera2_A_m, PoseCamera2_A_n);

fprintf(calib_para_doc, '\n%s\n', 'Stereo_A:');
for i = 1:1:PoseCamera2_A_m
    for j = 1:1:PoseCamera2_A_n
        if j == PoseCamera2_A_n
            fprintf(calib_para_doc, '%15.15f\n', PoseCamera2_A(i,j)); % 输出格式为浮点数,占用15位,保留15位小数
        else
            fprintf(calib_para_doc, '%15.15f\t', PoseCamera2_A(i,j));
        end
    end
end

fclose(calib_para_doc);