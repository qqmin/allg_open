%% �ۼ�Լ������
%% ����Լ���ڵ�����ǰ��
function [n, nr, crl] = CumulatedRestrainedList(nnj,rl)
kkl =0;
for jx = 1: nnj
    if rl(jx) == 0
        kkl = kkl +1;
        crl(jx) = kkl;
    end
end
n= kkl; %��Լ��λ����
for jx = 1:nnj
    if rl(jx) == 1
        kkl = kkl +1;
        crl(jx) = kkl;
    end
end
nr= kkl- n; %Լ��λ����
end