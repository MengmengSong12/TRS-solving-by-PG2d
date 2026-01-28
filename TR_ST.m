function [x,f,norm_g,k,k_f,k_gradshess,k_prods] = TR_ST(p,dimx,x,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta)
[f,g,H] = p.objective(x);
H=sparse(H);
%epsilon_g = max(epsilon_g*norm(g),1e-15);
k=0;k_f=1;k_gradshess=1;k_prods = 0;
norm_g = norm(g);
while norm_g > epsilon_g
    k=k+1;
    if k>K
        fprintf('Iteration number exceeds K\n');
        break;
    end
    % calculate Steihaug-Toint point
    v = zeros(dimx,1);
    r = -g;
    pp = r;
    k_st = 0;
    %epsilon_sub=max(epsilon_g*100/k,epsilon_g);
    epsilon_sub=norm_g*1e-4;
    while k_st<dimx
        k_prods = k_prods+1;
        Hpp = H*pp;
        pHp = pp'*Hpp;
        alpha = (r'*r)/pHp;
        v_plus = v+alpha*pp;
        if pHp<=0 || norm(v_plus) >= Delta
            a=pp'*pp;
            b=2*pp'*v;
            c=v'*v-Delta^2;
            v=v+((-b+sqrt(b^2-4*a*c))/(2*a))*pp;
            break;
        end
        v=v_plus;
        
        r_plus = r-alpha*Hpp;
        beta = (r_plus'*r_plus)/(r'*r);
        r=r_plus;
        pp_plus = r + beta*pp;
        pp=pp_plus;
        k_st=k_st+1;
        if norm(r)<epsilon_sub
            break;
        end
    end
    
    s=v;
    k_prods = k_prods+1;
    Hs=H*s;
    delta_q = -(g'*s)-1/2*s'*Hs;
    
    % end of Steihaug-Toint point calculation
    
    x_plus = x+s;
    
    f_plus = p.objective(x_plus);
    k_f = k_f+1;
    rho=(f-f_plus)/delta_q;
    fprintf('Step %d. : obj is %5.3e, norm of grad is %5.3e, Delta=%5.3e, delta_q=%5.3e, rho= %5.3e \n',  k, f, norm(g), Delta, delta_q, rho);
%     if delta_q<-1e-8
%         keyboard;
%     end
    if rho >= eta_1 && delta_q>0
        x = x_plus;
        [~,g_plus,H_plus] = p.objective(x_plus);
        k_gradshess=k_gradshess+1;
        f = f_plus;
        g = g_plus;
        H = sparse(H_plus);
        if rho >= eta_2 && Delta<1e9% Delta < 1e9 is removed, Delta can be unbounded from above
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
    if f<-1e6
            break;
    end
    if Delta <= epsilon_Delta
        break;
    end
end
end