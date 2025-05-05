%% 结构总节点刚度矩阵方程求解函数
function [d, r]= Statics_Response_Rod(n,nr,crl,K,F)
    
d(1:n+nr,1:1) = 0;
r(1:n+nr,1,1) = 0;    
b1 = (K(1:n,1:n));    
b2 = F(1:n);
%% 避免奇异值导致MATLAB产生错误的情况，采用“伪逆”来帮助我们解决这个问题——广义逆矩阵
%% matlab中的pinv函数
% d(1:n)= pinv(b1)*b2;                             % 计算非约束位移
d(1:n)= b1\b2;
%% 直接求解法： InvA=(A'A)\A'
% d(1 :n)= (b1'*b1)\b1'*b2;                           % 计算非约束位移 
%% SVD分解
% [u,s,v] = svd(b1);
% lemt = 1e-8;
% % 方法一：去掉极小奇异值
% s1 = diag(1 ./ s(s > lemt));
% s1(n,n) = 0;
% d(1:n)  = v*s1*u'*b2;

%% QR 分解——适用于稀疏矩阵
% [Q,R] = qr(b1);
% qrInvA  = (R'*R)\R'*Q';
% d(1 :n) = qrInvA*b2;
%% 支反力求解
r(n+1:n+nr)=-F(n + 1: n + nr) +K(n+ 1: n+nr, 1: n)*d(1:n) ;    % 计算文座反力
dx= d(crl(1:n+nr)) ; d = dx;                                   % 构建总节点位移向量
dx= r(crl(1:n+nr)) ; r = dx;                                   % 构建总约束反力向晕 
end