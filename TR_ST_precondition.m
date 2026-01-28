function [x,f,norm_g,k,k_f,k_gradshess,k_prods] = TR_ST_precondition(p,dimx,x,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta)
[f,g,H] = p.objective(x);
d=[-2 -1 0 1 2];
Bout = spdiags(H,d);
M = spdiags(Bout,d,dimx,dimx);
H=sparse(H);
M=sparse(M);
eigMin=min(eig(M));
eigMax=max(eig(M));
M=M-eigMin*eye(dimx)+(eigMax-eigMin)/2*eye(dimx);
M=M/norm(M,'fro');
%epsilon_g = max(epsilon_g*norm(g),1e-15);
k=0;k_f=1;k_gradshess=1;k_prods = 0;
norm_g = sqrt(g'*M*g);
while norm_g > epsilon_g
    k=k+1;
    if k>K
        fprintf('Iteration number exceeds K\n');
        break;
    end
    % calculate Steihaug-Toint point
    v = zeros(dimx,1);
    r = -g;
    rt = M\r;
    pp = rt;
    k_st = 0;
    epsilon_sub = max(epsilon_g*100/K,epsilon_g);
    while k_st<=dimx
        k_prods = k_prods+1;
        Hpp = H*pp;
        pHp = pp'*Hpp;
        alpha = (r'*rt)/pHp;
        v_plus = v+alpha*pp;
        if pHp<=0 || v_plus'*M*v_plus >= Delta^2
            a=pp'*M*pp;
            b=2*pp'*M*v;
            c=v'*M*v-Delta^2;
            v=v+((-b+sqrt(b^2-4*a*c))/(2*a))*pp;
            break;
        end
        v=v_plus;
        
        r_plus = r-alpha*Hpp;
        r_plus_t=M\r_plus;
        beta = (r_plus'*r_plus_t)/(r'*rt);
        r=r_plus;
        rt=r_plus_t;
        pp_plus = rt + beta*pp;
        pp=pp_plus;
        k_st=k_st+1;
        if sqrt(r'*M*r)<epsilon_sub
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
        Bout = spdiags(H_plus,d);
        M = spdiags(Bout,d,dimx,dimx);
        M=sparse(M);
        eigMin=min(eig(M));
        eigMax=max(eig(M));
        if eigMin/eigMax<1e-6
            M=M-(eigMin-(eigMin+eigMax)/2)*eye(dimx);
        end
        M=M/norm(M,'fro');
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