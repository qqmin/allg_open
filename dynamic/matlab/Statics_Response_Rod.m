%% �ṹ�ܽڵ�նȾ��󷽳���⺯��
function [d, r]= Statics_Response_Rod(n,nr,crl,K,F)
    
d(1:n+nr,1:1) = 0;
r(1:n+nr,1,1) = 0;    
b1 = (K(1:n,1:n));    
b2 = F(1:n);
%% ��������ֵ����MATLAB�����������������á�α�桱���������ǽ��������⡪�����������
%% matlab�е�pinv����
% d(1:n)= pinv(b1)*b2;                             % �����Լ��λ��
d(1:n)= b1\b2;
%% ֱ����ⷨ�� InvA=(A'A)\A'
% d(1 :n)= (b1'*b1)\b1'*b2;                           % �����Լ��λ�� 
%% SVD�ֽ�
% [u,s,v] = svd(b1);
% lemt = 1e-8;
% % ����һ��ȥ����С����ֵ
% s1 = diag(1 ./ s(s > lemt));
% s1(n,n) = 0;
% d(1:n)  = v*s1*u'*b2;

%% QR �ֽ⡪��������ϡ�����
% [Q,R] = qr(b1);
% qrInvA  = (R'*R)\R'*Q';
% d(1 :n) = qrInvA*b2;
%% ֧�������
r(n+1:n+nr)=-F(n + 1: n + nr) +K(n+ 1: n+nr, 1: n)*d(1:n) ;    % ������������
dx= d(crl(1:n+nr)) ; d = dx;                                   % �����ܽڵ�λ������
dx= r(crl(1:n+nr)) ; r = dx;                                   % ������Լ���������� 
end