function [x,f,norm_g,k,k_f,k_gradshess,Delta] = TR_Cauchy(p,dimx,x,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta)
[f,g,H] = p.objective(x);
H=sparse(H);
%epsilon_g = max(epsilon_g*norm(g),1e-15);
s = zeros(dimx,1);
Hg=H*g;
k=0;k_f=1; k_gradshess=1;
norm_g = norm(g);
while norm_g > epsilon_g
    k=k+1;
    if k>K
        fprintf('Iteration number exceeds K\n');
        break;
    end
    % Cauchy point first
    
    a0 = g'*Hg/2;
    b0 = -g'*g;
    if a0 < 0
        t = Delta/sqrt(-b0);
    else
        t = min(Delta/sqrt(-b0),-b0/(2*a0));
    end
    s = -t*g;
    Hs=-t*Hg;
    delta_q = -(t*b0)-1/2*s'*Hs;
    % end of Cauchy point calculation
    x_plus = x+s;
    f_plus = p.objective(x_plus);
    k_f = k_f+1;
    rho=(f-f_plus)/delta_q;
    fprintf('Step %d. : obj is %5.3e, norm of grad is %5.3e, Delta=%5.3e, delta_q=%5.3e, rho= %5.3e \n',  k, f, norm(g), Delta, delta_q, rho);
    if rho >= eta_1 && delta_q>0
        x = x_plus;
        [~,g_plus,H_plus] = p.objective(x_plus);
        k_gradshess=k_gradshess+1;
        f = f_plus;
        g = g_plus;
        H = sparse(H_plus);
        Hg=H*g;
        if rho >= eta_2 && Delta<1e9 % If Delta < 1e9 is removed, Delta can be unbounded from above
            Delta = gamma_2*Delta;
        end
    else
        Delta = gamma_1*Delta;
    end
    %     %solve TRS
    %     [s] = TRS(g, H, Delta, k_TRS); % solve TRS, k_TRS is the step request
    %     x_plus = x + s;
    %     delta_q = -g'*s-1/2*s'*H*s;
    %     [f_plus,g,H] = p.objective(x_plus);
    norm_g = norm(g);
    if f<-1e5
            break;
    end
    if Delta <= epsilon_Delta
        break;
    end
end
end