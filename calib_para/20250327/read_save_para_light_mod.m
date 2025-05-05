%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 精简可python版本
% 读取calibrationSession.mat中指定标定参数并保存至文本calib_para_light.txt中
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;

calib_data = load('calibrationSession.mat');

calib_para_doc = fopen('calib_para_light.txt', 'wt');

ImageSize = calib_data.calibrationSession.CameraParameters.CameraParameters1.ImageSize;
fprintf(calib_para_doc, '%d\n', ImageSize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 保存left参数 K & Distortion
K_left = calib_data.calibrationSession.CameraParameters.CameraParameters1.K;
RadialDist_left = calib_data.calibrationSession.CameraParameters.CameraParameters1.RadialDistortion;
TangenDist_left = calib_data.calibrationSession.CameraParameters.CameraParameters1.TangentialDistortion;

Dist_left(1,1) = RadialDist_left(1,1);
Dist_left(1,2) = RadialDist_left(1,2);
Dist_left(1,3) = TangenDist_left(1,1);
Dist_left(1,4) = TangenDist_left(1,2);
Dist_left(1,5) = 0.0;

% 保存left参数 K & Distortion
fprintf(calib_para_doc, '%15.15f\n', K_left(1,1)); % 输出格式为浮点数,占用15位,保留15位小数
fprintf(calib_para_doc, '%15.15f\n', K_left(2,2));
fprintf(calib_para_doc, '%15.15f\n', K_left(1,2));
fprintf(calib_para_doc, '%15.15f\n', K_left(1,3));
fprintf(calib_para_doc, '%15.15f\n', K_left(2,3));
fprintf(calib_para_doc, '%15.15f\n', Dist_left);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 保存right参数 K & Distortion
K_right = calib_data.calibrationSession.CameraParameters.CameraParameters2.K;
RadialDist_right = calib_data.calibrationSession.CameraParameters.CameraParameters2.RadialDistortion;
TangenDist_right = calib_data.calibrationSession.CameraParameters.CameraParameters2.TangentialDistortion;

Dist_right(1,1) = RadialDist_right(1,1);
Dist_right(1,2) = RadialDist_right(1,2);
Dist_right(1,3) = TangenDist_right(1,1);
Dist_right(1,4) = TangenDist_right(1,2);
Dist_right(1,5) = 0.0;

% 保存right参数 K & Distortion
fprintf(calib_para_doc, '%15.15f\n', K_right(1,1)); % 输出格式为浮点数,占用15位,保留15位小数
fprintf(calib_para_doc, '%15.15f\n', K_right(2,2));
fprintf(calib_para_doc, '%15.15f\n', K_right(1,2));
fprintf(calib_para_doc, '%15.15f\n', K_right(1,3));
fprintf(calib_para_doc, '%15.15f\n', K_right(2,3));
fprintf(calib_para_doc, '%15.15f\n', Dist_right);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 显示并保存stereo参数 PoseCamera2_A
PoseCamera2_R = calib_data.calibrationSession.CameraParameters.PoseCamera2.R;
PoseCamera2_T = calib_data.calibrationSession.CameraParameters.PoseCamera2.Translation;

fprintf(calib_para_doc, '%15.15f\n', PoseCamera2_R); % 输出格式为浮点数,占用15位,保留15位小数
fprintf(calib_para_doc, '%15.15f\n', PoseCamera2_T);

fclose(calib_para_doc);