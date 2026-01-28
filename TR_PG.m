function [x,f,ng,k,k_f,k_gradshess,H] = TR_PG(p,dimx,x,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta)
[f,g,H] = p.objective(x);
H=sparse(H);
%epsilon_g = max(epsilon_g*norm(g),1e-15);
k=0;k_f=1; k_gradshess=1;
ng = norm(g);
Lbar=0.01;
while ng > epsilon_g
    %stopping critiria for solving TRS using double-start PG
    epsilon_sub=ng*1e-4;
    k=k+1;
    if k>K
        fprintf('Iteration number exceeds K\n');
        break;
    end
    % Cauchy point first
    Hg=H*g;
    a0 = g'*Hg/2;
    b0 = -g'*g;
    if a0 < 0
        t = Delta/sqrt(-b0);
    else
        t = min(Delta/sqrt(-b0),-b0/(2*a0));
    end
    s = -t*g;
    Hs1=-t*Hg;
    %delta_q = -(g'*s)-1/2*s'*Hs1;
    % end of Cauchy point calculation
    %start of PG for globally solving TRS
    s1 = s;
    s2 = 1e-10*randn(dimx,1);
    %s2 = zeros(dimx,1);
    temp = sqrt(s1'*s1+s2'*s2);
    if temp > Delta
        s1 = s1/temp*Delta;
        Hs1 = Hs1/temp*Delta;
        s2 = s2/temp*Delta;
    end
    Hs2 = H*s2;
    a0 = min(temp,Delta);
    grads1 = Hs1+g;
    grads2 = Hs2;
    a1 = s1/a0;
    Ha1 = Hs1/a0;
    a2 = s2/a0;
    Ha2 = Hs2/a0;
    Hgrads1 = H*grads1;
    Hgrads2 = H*grads2;
    temp = grads1'*a1+grads2'*a2;
    b1 = grads1-temp*a1;
    Hb1 = Hgrads1-temp*Ha1;
    b2 = grads2-temp*a2;
    Hb2 = Hgrads2-temp*Ha2;
    temp = sqrt(b1'*b1+b2'*b2);
    b1 = b1/temp;
    b2 = b2/temp;
    Hb1 = Hb1/temp;
    Hb2 = Hb2/temp;
    
    H11  = a1'*Ha1+a2'*Ha2;
    H12 = a1'*Hb1+a2'*Hb2;
    H22 = b1'*Hb1+b2'*Hb2;
    delta = (H11+H22)^2-4*(H11*H22-H12^2);

    stepsize = 2/(abs(H11+H22)+sqrt(delta));
    for ii = 1:k
        s1 = s1 - stepsize*grads1;
        s2 = s2 - stepsize*grads2;
        
        Hs1 = Hs1 - stepsize*Hgrads1;
        Hs2 = Hs2 - stepsize*Hgrads2;
        temp = sqrt(s1'*s1+s2'*s2);
        if temp > Delta
            s1 = s1/temp*Delta;
            s2 = s2/temp*Delta;
            Hs1 = Hs1/temp*Delta;
            Hs2 = Hs2/temp*Delta;
        end
        
        grads1 = Hs1 + g;
        grads2 = Hs2;
        Hgrads1 = H*grads1;
        Hgrads2 = H*grads2;
        
        
        %for terminiation check
        deltas=(s1'*s1+s2'*s2)^0.5;
        deltagrads=(grads1'*grads1+grads2'*grads2)^0.5;
        
        if (deltagrads <epsilon_sub && deltas<=Delta) || (grads1'*s1+grads2'*s2 <-(1-epsilon_sub)*deltagrads*deltas && deltas>(1-epsilon_sub)*Delta)
            break;
        end
        % for the calculation of stepsize
        a0 = min(temp,Delta);
        a1 = s1/a0;
        Ha1 = Hs1/a0;
        a2 = s2/a0;
        Ha2 = Hs2/a0;
        
        temp = grads1'*a1+grads2'*a2;
        b1 = grads1-temp*a1;
        Hb1 = Hgrads1-temp*Ha1;
        
        b2 = grads2-temp*a2;
        Hb2 = Hgrads2-temp*Ha2;
        temp = sqrt(b1'*b1+b2'*b2);
        
        b1 = b1/temp;
        b2 = b2/temp;
        Hb1 = Hb1/temp;
        Hb2 = Hb2/temp;
        
        H11  = a1'*Ha1+a2'*Ha2;
        H12 = a1'*Hb1+a2'*Hb2;
        H22 = b1'*Hb1+b2'*Hb2;
        delta = (H11+H22)^2-4*(H11*H22-H12^2);
        Lk = ((H11+H22)+sqrt(delta))/2;
        if Lk >= Lbar
            stepsize = 1/Lk;
        else
            stepsize = 1/Lbar; 
        end
    end
    s = s1;
    delta_q = -(g'*s1)-1/2*s1'*Hs1;
    %if delta_q<=1e-30
        %break;
    %end
    if norm(s2)>1e-8
        t1 = (-s1'*s2+sqrt((s1'*s2)^2-s2'*s2*(s1'*s1-Delta^2)))/(s2'*s2);
        t2 = (-s1'*s2-sqrt((s1'*s2)^2-s2'*s2*(s1'*s1-Delta^2)))/(s2'*s2);
        st1 = s1+t1*s2;
        st2 = s1+t2*s2;
        delta_qst1 = -(g'*st1)-1/2*st1'*(Hs1+t1*Hs2);
        delta_qst2 = -(g'*st2)-1/2*st2'*(Hs1+t2*Hs2);
        if delta_qst1 > delta_qst2 && delta_qst1 > delta_q
            s = st1;
            delta_q = delta_qst1;
        elseif delta_qst2 > delta_q
            s = st2;
            delta_q = delta_qst2;
        end   
    end
    x_plus = x+s;
    [f_plus] = p.objective(x_plus);
    k_f=k_f+1;
    rho=(f-f_plus)/delta_q;
    fprintf('Step %d. : obj is %5.3e, norm of grad is %5.3e, Delta=%5.3e, delta_q=%5.3e, rho= %5.3e \n', k, f, norm(g), Delta, delta_q, rho);
    if rho >= eta_1 && delta_q>0
        x = x_plus;
        [~,g_plus,H_plus] = p.objective(x_plus);
        k_gradshess=k_gradshess+1;
        f = f_plus;
        g = g_plus;
        H = sparse(H_plus);
        if rho >= eta_2 && Delta<1e9
            Delta = gamma_2*Delta;
        end
    else
        Delta = gamma_1*Delta;
    end
    if Delta <= epsilon_Delta
        break;
    end
    if f<-1e6
            break;
    end
    ng = norm(g);
end


%     %solve TRS
%     [s] = TRS(g, H, Delta, k_TRS); % solve TRS, k_TRS is the step request
%     x_plus = x + s;
%     delta_q = -g'*s-1/2*s'*H*s;
%     [f_plus,g,H] = p.objective(x_plus);
%
end