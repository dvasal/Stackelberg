function [ V0,V1 ] = reward2go_0_inf( pi1v,F1_0,F1_1,u)

f = @(x) interp1(pi1v,u,x) ;

V0 = f(F1_0);
V1 = f(F1_1);

end
