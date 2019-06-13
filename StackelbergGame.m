clc
clear

% Let's say Player 1 is the follower and Player 2 is the Leader
eps = 1e-3;
T=20; % horizon

% Leaders' Payoffs
Rl_L(1,1) = 2;
Rl_L(1,2) = 4;
Rl_L(2,1) = 1;
Rl_L(2,2) = 3;

Rl_H(1,1) = 3 ;
Rl_H(1,2) = 2;
Rl_H(2,1) = 0;
Rl_H(2,2) = 1;

%Follower's Payoffs

Rf_L(1,1) = 1;
Rf_L(1,2) = 0;
Rf_L(2,1) = 0;
Rf_L(2,2) = 2;

Rf_H(1,1) = 2;        % Player 2 has no private type
Rf_H(1,2) = 0;
Rf_H(2,1) = 1;
Rf_H(2,2) = 1;

% discount factor for infinite horizon reward.
delI = 1; % discount factor for equilibrium update. It dampens the step size

N=60; %resolution for pi space
pi1v=(0:N)*(1/N);
%pi2v=(0:N)*(1/N);

results=[];
%Strategies
p1Lm=0.5*ones(1,N+1);
p1Hm=0.5*zeros(1,N+1);
p2m=0.5*ones(1,N+1);
%Utilities-to-go
u1Lm=zeros(1,N+1);
u1Hm=zeros(1,N+1);
u2m=zeros(1,N+1);

delD = 0.6;

BR_f = zeros(N+1,2);
Eq = zeros(N+1,3);
t=0;
err_u=1;
while t<=T && err_u>1e-4
    t=t+1;
    delD
    % For every t
    t % DISPLAY
    p1Lm_n=zeros(1,N+1);
    p1Hm_n=zeros(1,N+1);
    p2m_n=zeros(1,N+1);
    
    u1Lm_n=zeros(1,N+1);
    u1Hm_n=zeros(1,N+1);
    u2m_n=zeros(1,N+1);
 
    for i1=1:N+1       % For every pi
        
        pi1=pi1v(i1);
        
        countp2=0;
        for p2 = 0:1/N:1
            countp2 = countp2+1;
            % Calculate BR^f(p2)
            p1L = p1Lm(i1);
            p1H = p1Hm(i1);
            
            err = 1;
            count = 0;
            
            while err > 1e-4 && count <=1e3        % Solve the FP equation
                count = count+1;
                
                % check for discontinuitites
                if (pi1*(1-p1H) + (1-pi1)*(1-p1L)) == 0
                    F1_0 = pi1;
                else
                    F1_0 = pi1*(1-p1H)/(pi1*(1-p1H) + (1-pi1)*(1-p1L));
                end
                
                if (pi1*p1H + (1-pi1)*p1L) == 0
                    F1_1 = pi1;
                else
                    F1_1 = pi1*p1H/(pi1*p1H + (1-pi1)*p1L);
                end
                
                V1L0 = interp1(pi1v,u1Lm,F1_0);   
                V1L1 = interp1(pi1v,u1Lm,F1_1);
                V1H0 = interp1(pi1v,u1Hm,F1_0);
                V1H1 = interp1(pi1v,u1Hm,F1_1);     
                
                % User 1 state L
                u1L_0 = (1- p2)*Rf_L(1,1) + (p2)*Rf_L(1,2) + delD*V1L0;
                u1L_1 = (1- p2)*Rf_L(2,1) + (p2)*Rf_L(2,2) + delD*V1L1  ;
                u1L_s = (1-p1L) * u1L_0 + p1L * u1L_1;
                
                phi1L_0 = delI*abs(u1L_0 - u1L_s);
                phi1L_1 = delI*abs(u1L_1 - u1L_s);
                if u1L_1>=u1L_s
                    p1L_n = p1L + phi1L_1/(phi1L_0 + phi1L_1);
                else
                    p1L_n = p1L - phi1L_0/(phi1L_0 + phi1L_1);
                end
                % User 1 , state H
                u1H_0 = (1- p2)*Rf_H(1,1) + (p2)*Rf_H(1,2) + delD*V1H0 ;
                u1H_1 = (1- p2)*Rf_H(2,1) + (p2)*Rf_H(2,2)  + delD*V1H1 ;
                u1H_s =  (1-p1H)* u1H_0 + (p1H)* u1H_1;
                
                phi1H_0 = delI*abs(u1H_0 - u1H_s);
                phi1H_1 = delI*abs(u1H_1 - u1H_s);
                %p1H_n = (p1H + phi1H_1)/ (1 + phi1H_0 + phi1H_1 );
                if u1H_1>=u1H_s
                    p1H_n = p1H + phi1H_1/(phi1H_0 + phi1H_1);
                else
                    p1H_n = p1H - phi1H_0/(phi1H_0 + phi1H_1);
                end
                
                err = norm([p1L p1H] - [p1L_n p1H_n]);
                
                p1L = p1L_n;
                p1H = p1H_n;
                
            end
            BR_f(countp2,1) = p1L;
            BR_f(countp2,2) = p1H;
        end
        
        countp2 = 0;
        for p2 = 0:1/N:1
            countp2 = countp2+1;
            p1L = BR_f(countp2,1);
            p1H = BR_f(countp2,2);
            if (pi1*(1-p1H) + (1-pi1)*(1-p1L)) == 0
                F1_0 = pi1;
            else
                F1_0 = pi1*(1-p1H)/(pi1*(1-p1H) + (1-pi1)*(1-p1L));
            end
            
            if (pi1*p1H + (1-pi1)*p1L) == 0
                F1_1 = pi1;
            else
                F1_1 = pi1*p1H/(pi1*p1H + (1-pi1)*p1L);
            end
                
                V20 =  interp1(pi1v,u2m,F1_0);
                V21 =  interp1(pi1v,u2m,F1_1);

            u2_0 = (1-pi1)*((1-p1L)*Rl_L(1,1) + p1L*Rl_L(2,1)) + pi1*((1-p1H)*Rl_H(1,1) + p1H*Rl_H(2,1)) ...
                +  delD*[ (1-(1-pi1)*p1L - pi1*p1H)* V20 + ((1-pi1)*p1L + pi1*p1H)* V21 ] ;
            u2_1 = (1-pi1)*((1-p1L)*Rl_L(1,2) + p1L*Rl_L(2,2)) + pi1*((1-p1H)*Rl_H(1,2) + p1H*Rl_H(2,2)) ...
                +  delD*[ (1-(1-pi1)*p1L - pi1*p1H)* V20 + ((1-pi1)*p1L + pi1*p1H)* V21 ] ;
            
            u2_s = (1-p2) * u2_0 + p2 * u2_1;
            
            if (u2_s>u2_0-eps) && (u2_s>u2_1-eps)
                
                Eq(i1,:) = [p1L p1H p2];
                break;
                
            end
            
        end
        
        p1L = Eq(i1,1);
        p1H = Eq(i1,2);
        p2 = Eq(i1,3);
        
        %%
        
        if (pi1*(1-p1H) + (1-pi1)*(1-p1L)) == 0
            F1_0 = pi1;
        else
            F1_0 = pi1*(1-p1H)/(pi1*(1-p1H) + (1-pi1)*(1-p1L));
        end
        
        if (pi1*p1H + (1-pi1)*p1L) == 0
            F1_1 = pi1;
        else
            F1_1 = pi1*p1H/(pi1*p1H + (1-pi1)*p1L);
        end
        
                V1L0 = interp1(pi1v,u1Lm,F1_0);   
                V1L1 = interp1(pi1v,u1Lm,F1_1);
                V1H0 = interp1(pi1v,u1Hm,F1_0);
                V1H1 = interp1(pi1v,u1Hm,F1_1);     
                V20 =  interp1(pi1v,u2m,F1_0);
                V21 =  interp1(pi1v,u2m,F1_1);
        
        % User 1 state L
        u1L_0 = (1- p2)*Rf_L(1,1) + (p2)*Rf_L(1,2) + delD*V1L0;
        u1L_1 = (1- p2)*Rf_L(2,1) + (p2)*Rf_L(2,2) + delD*V1L1  ;
        
        
        % User 1 , state H
        u1H_0 = (1- p2)*Rf_H(1,1) + (p2)*Rf_H(1,2) + delD*V1H0 ;
        u1H_1 = (1- p2)*Rf_H(2,1) + (p2)*Rf_H(2,2)  + delD*V1H1 ;
        
        % User 2
        u2_0 = (1-pi1)*((1-p1L)*Rl_L(1,1) + p1L*Rl_L(2,1)) + pi1*((1-p1H)*Rl_H(1,1) + p1H*Rl_H(2,1)) ...
            +  delD*[ (1-(1-pi1)*p1L - pi1*p1H)* V20 + ((1-pi1)*p1L + pi1*p1H)* V21 ] ;
        u2_1 = (1-pi1)*((1-p1L)*Rl_L(1,2) + p1L*Rl_L(2,2)) + pi1*((1-p1H)*Rl_H(1,2) + p1H*Rl_H(2,2)) ...
            +  delD*[ (1-(1-pi1)*p1L - pi1*p1H)* V20 + ((1-pi1)*p1L + pi1*p1H)* V21 ] ;
        %p2L_n = p1L_n; % force symmetric eq
        
        u1L = (1-p1L) * u1L_0 + p1L * u1L_1;
        u1H = (1-p1H)* u1H_0 +  (p1H)* u1H_1;
        u2 = (1-p2) * u2_0 + p2 * u2_1;
        
        p1Lm_n(i1)=p1L;
        p1Hm_n(i1)=p1H;
        p2m_n(i1)=p2;
        
        u1Lm_n(i1)=u1L;
        u1Hm_n(i1)=u1H;
        u2m_n(i1)=u2;
        
    end
    
    err_p = norm([p1Lm p1Hm p2m] - [p1Lm_n p1Hm_n p2m_n ]);
    err_u = norm([u1Lm u1Hm u2m] - [u1Lm_n u1Hm_n u2m_n ]);
    [err_p err_u] % DISPLAY
    p1Lm=p1Lm_n;
    p1Hm=p1Hm_n;
    p2m=p2m_n;
    
    u1Lm=u1Lm_n;
    u1Hm=u1Hm_n;
    u2m=u2m_n;
    
end

figure
plot(pi1v,p1Lm)
grid on
xlabel('\pi(1)')
ylabel('p^{f,0}')
title('Probability of follower taking action 1 when state is low')

figure
plot(pi1v,p1Hm)
grid on
xlabel('\pi(1)')
ylabel('p^{f,1}')
title('Probability of follower taking action 1 when state is high')

figure
plot(pi1v,p2m)
grid on
xlabel('\pi(1)')
ylabel('p^l')
title('Probability of leader taking action 1')

figure
plot(pi1v,u1Lm)
grid on
xlabel('\pi(1)')
ylabel('u^{f,0}')
title('Utility of the follower when the state is low')

figure
plot(pi1v,u1Hm)
grid on
xlabel('\pi(1)')
ylabel('u^{f,1}')
title('Utility of the follower when the state is high')

figure
plot(pi1v,u2m)
grid on
xlabel('\pi(1)')
ylabel('u^l')
title('Utility of the leader')