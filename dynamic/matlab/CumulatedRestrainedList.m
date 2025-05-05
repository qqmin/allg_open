%% 累计约束表函数
%% 将非约束节点排在前方
function [n, nr, crl] = CumulatedRestrainedList(nnj,rl)
kkl =0;
for jx = 1: nnj
    if rl(jx) == 0
        kkl = kkl +1;
        crl(jx) = kkl;
    end
end
n= kkl; %非约束位移数
for jx = 1:nnj
    if rl(jx) == 1
        kkl = kkl +1;
        crl(jx) = kkl;
    end
end
nr= kkl- n; %约束位移数
end