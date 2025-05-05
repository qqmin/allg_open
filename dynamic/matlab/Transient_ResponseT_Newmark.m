%% 以下函数用Newmark法计算结构的动响应．
function [acc_1,vel_1,dsp_1] = Transient_ResponseT_Newmark(kk,cc,mm,fd,dt,dsp_0,vel_0,acc_0)
%% -------------------------------------------------------------------------------
% Purpose:
%     The function subroutine TransResp4.m calculates transient response of  
%     a structural system using Newmark integration scheme.
%  Synopsis: 
%     [ acc, vel, dsp] =TransResp4 (kk, cc, mm, fd, bcdof, nt, dt, q0, dq0)
%  Variable Description: 
%   Input parameters
%       kk, cc, mm - stiffness, damping and mass matrixes 
%       fd - Input or forcing influence matrix
%       bcdof - Boundary condition dofs vector 
%       nt - Number of time steps 
%       dt - Time step size 
%       q0, dq0 - Initial condition vectors 
%   Output parameters
%     acc - Acceleration response 
%     vel - Velocity response 
%     dsp - Displacement response
%% -------------------------------------------------------------------------------

alpha = 0.5;              % select the parameters
beta  = 0.5;              % select the parameters
ekk   = kk + mm/(alpha*dt^2) + cc*beta/(alpha*dt);                  % compute the effective stiffness matrix 

cfm = dsp_0/(alpha*dt^2)    + vel_0/(alpha*dt)     + acc_0*(0.5/alpha-1); 
cfc = dsp_0*beta/(alpha*dt) + vel_0*(beta/alpha-1) + acc_0*(0.5*beta/alpha-1)*dt;
efd = fd  + mm*cfm + cc*cfc;           % compute the effective force vector

% dsp_1 = pinv(ekk)*efd;
dsp_1 = ekk\efd;
acc_1 = (dsp_1 - dsp_0)/(alpha* dt^2) - vel_0/(alpha*dt) - acc_0*(0.5/alpha-1);
vel_1 =  vel_0 + acc_0*(1-beta)*dt + acc_1*beta*dt;


% if cc(1,1)==0 
%     disp ('The transient response results of undampin_g system') 
% else
%     disp('The transient response results of damping system') 
% end
% disp('The method is Newmark integration') 
end
% -------------------------------------------------------------------------------
% The end
% -------------------------------------------------------------------------------


