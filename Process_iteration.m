clc;
clear;
%problist = secup(struct('type', 'u', 'mindim', 100, 'maxdim', 2000));
%problist={'FLETCHCR'};
%problist={'PATTERNLS'};
%problist={'PENALTY3'};
problist={'EDENSCH'};
%problist={'LUKSAN11LS'};
%problist={'SPINLS'};
%problist={'VARDIM'};
%problist={'COATING','EDENSCH','EXTROSNB','FLETCHCR','GENROSE','KSSLS','LRW8A','LUKSAN11LS',...
 %   'LUKSAN15LS','LUKSAN16LS','LUKSAN17LS','LUKSAN21LS','LUKSAN22LS','MANCINO',...
 %   'MNISTS0LS','MSQRTALS','MSQRTBLS','PATTERNLS','PENALTY1','PENALTY2','PENALTY3',...
 %   'QING','SENSORS','SPIN2LS','SPINLS','VARDIM'};
%problist={'ARGLINC','ARGTRIGLS','LRA9A','MNISTS5LS'};
a=clock;
epsilon_Delta = 1e-10;
for ip = 1: length(problist)
%for ip = 1: 1
    pname = problist{ip};
    p = macup(pname);  % make a CUTEst problem
    x0=p.x0;
    dimx = length(p.x0);
    fprintf('\n%d. Solve %s:\n', ip, pname);
    %%TR method setting
    epsilon_g = 1e-2;
    eta_1 = 0.01;
    eta_2 = 0.95;
    gamma_1 = 0.5;
    gamma_2 = 2;
    K = 10;
    Delta = 4;
    
    %%%data collection, data in each time
    
    steps = 10;
    Process_iteration_doublePG=zeros(steps+1,1);
    Process_iteration_doublePG_extended=zeros(steps+1,1);
    Process_iteration_PG2d_extended=zeros(steps+1,1);
    
    fprintf('------------------START of TR_Cauchy------------------\n');
    [x_C,f_C,norm_g_C,step_C,k_f_C,k_gradshess_C,Delta] = TR_Cauchy(p,dimx,x0,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta);
    fprintf('------------------END of TR_Cauchy------------------\n');

    if norm_g_C<1e-1
        continue;
    end
    [~,g,H] = p.objective(x_C);
    
    % the TRS is min 1/2*x'*H*x+g'*x s.t. x'*x<=Delta.
    %the same initial
    s10=zeros(dimx,1);
    s20=randn(dimx,1);
    s20=s20/norm(s20)*Delta;
    
    
    
    %%% iterations of double start PG+constant stepsizes(1/L)
    s1=s10;
    s2=s20;
    obj1=1/2*s1'*H*s1+g'*s1;
    obj2=1/2*s2'*H*s2+g'*s2;
    Process_iteration_doublePG(1)=min(obj1,obj2);
    
    grads1=H*s1+g;
    grads2=H*s2+g;
    stepsize = 1/max(abs(eig(H)));
    for jj=1:steps
        s1=s1-stepsize*grads1;
        s2=s2-stepsize*grads2;
        grads1=H*s1+g;
        grads2=H*s2+g;
        temp1=norm(s1);
        temp2=norm(s2);
        if temp1>Delta
            s1=s1/temp1*Delta;
        end
        if temp2>Delta
            s2=s2/temp2*Delta;
        end
        obj1=1/2*s1'*H*s1+g'*s1;
        obj2=1/2*s2'*H*s2+g'*s2;
        Process_iteration_doublePG(jj+1)=min(obj1,obj2);
    end
    
    
    
    
    %%% iterations of double start PG+extended stepsizes
    s1=s10;
    s2=s20;
    obj1=1/2*s1'*H*s1+g'*s1;
    obj2=1/2*s2'*H*s2+g'*s2;
    Process_iteration_doublePG_extended(1)=min(obj1,obj2);
    
    grads1=H*s1+g;
    grads2=H*s2+g;
    
    Lbar=0.01;
    % for the calculation of stepsize
    if s1'*s1<1e-30
        stepsize1=1/(grads1'*H*grads1/(grads1'*grads1));
    else
    d1 = grads1 - s1'*grads1/(s1'*s1)*s1;
    d1 = d1/norm(d1);
    d2 = s1/norm(s1);
    H11 = d1'*H*d1;
    H12 = d1'*H*d2;
    H22 = d2'*H*d2;
    Lk = ((H11+H22)+sqrt((H11+H22)^2-4*(H11*H22-H12^2)))/2;
    if Lk>=Lbar
        stepsize1=1/Lk;
    else
        stepsize1=1/Lbar;
    end
    end

    if s2'*s2<1e-30
        stepsize2=1/(grads2'*H*grads2/(grads2'*grads2));
    else
    d1 = grads2 - s2'*grads2/(s2'*s2)*s2;
    d1 = d1/norm(d1);
    d2 = s2/norm(s2);
    H11 = d1'*H*d1;
    H12 = d1'*H*d2;
    H22 = d2'*H*d2;
    Lk = ((H11+H22)+sqrt((H11+H22)^2-4*(H11*H22-H12^2)))/2;
    if Lk>=Lbar
        stepsize2=1/Lk;
    else
        stepsize2=1/Lbar;
    end
    end
    
    for jj=1:steps
       
        s1=s1-stepsize1*grads1;
        s2=s2-stepsize2*grads2;
        
        temp1=norm(s1);
        temp2=norm(s2);
        if temp1>Delta
            s1=s1/temp1*Delta;
        end
        if temp2>Delta
            s2=s2/temp2*Delta;
        end
        obj1=1/2*s1'*H*s1+g'*s1;
        obj2=1/2*s2'*H*s2+g'*s2;
        Process_iteration_doublePG_extended(jj+1)=min(obj1,obj2);
        % for the calculation of stepsize
        grads1=H*s1+g;
        grads2=H*s2+g;
        if s1'*s1<1e-30
        stepsize1=1/(grads1'*H*grads1/(grads1'*grads1));
        else
        d1 = grads1 - s1'*grads1/(s1'*s1)*s1;
        d1 = d1/norm(d1);
        d2 = s1/norm(s1);
        H11 = d1'*H*d1;
        H12 = d1'*H*d2;
        H22 = d2'*H*d2;
        Lk = ((H11+H22)+sqrt((H11+H22)^2-4*(H11*H22-H12^2)))/2;
        if Lk>=Lbar
            stepsize1=1/Lk;
        else
        stepsize1=1/Lbar;
        end
        end

        if s2'*s2<1e-30
        stepsize2=1/(grads2'*H*grads2/(grads2'*grads2));
        else
        d1 = grads2 - s2'*grads2/(s2'*s2)*s2;
        d1 = d1/norm(d1);
        d2 = s2/norm(s2);
        H11 = d1'*H*d1;
        H12 = d1'*H*d2;
        H22 = d2'*H*d2;
        Lk = ((H11+H22)+sqrt((H11+H22)^2-4*(H11*H22-H12^2)))/2;
        if Lk>=Lbar
            stepsize2=1/Lk;
        else
            stepsize2=1/Lbar;
        end
        end
    end
    
    %%% iterations of PG_2d
    s1=s10;
    s2=s20;
    s = s1;
    Hs1=H*s1;
    Hs2=H*s2;
    delta_q = -(g'*s1)-1/2*s1'*Hs1;
    if norm(s2)>1e-30
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
    Process_iteration_PG2d_extended(1)=-delta_q;
    
    grads1=Hs1+g;
    grads2=Hs2;
    Hgrads1=H*grads1;
    Hgrads2=H*grads2;
    
    %for terminiation check
    temp = sqrt(s1'*s1+s2'*s2);
    deltas=(s1'*s1+s2'*s2)^0.5;
    deltagrads=(grads1'*grads1+grads2'*grads2)^0.5;
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
    Lk = ((H11+H22)+sqrt((H11+H22)^2-4*(H11*H22-H12^2)))/2;
    if Lk>=Lbar
        stepsize=1/Lk;
    else
        stepsize=1/Lbar;
    end
    for jj=1:steps
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
%         
         grads1 = Hs1 + g;
         grads2 = Hs2;
         Hgrads1 = H*grads1;
         Hgrads2 = H*grads2;
        
        
        %for terminiation check
        deltas=(s1'*s1+s2'*s2)^0.5;
        deltagrads=(grads1'*grads1+grads2'*grads2)^0.5;
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
        Lk = ((H11+H22)+sqrt((H11+H22)^2-4*(H11*H22-H12^2)))/2;
        if Lk>=Lbar
            stepsize=1/Lk;
        else
            stepsize=1/Lbar;
        end
        %obtain an s using s1 and s2
        s = s1;
        delta_q = -(g'*s1)-1/2*s1'*Hs1;
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
        Process_iteration_PG2d_extended(jj+1)=-delta_q;
    end
    %%% save eps

    %figure('Position',[500 500 500 500]);

    figure('Position',[500 500 500 500]);

    plot(0:steps,Process_iteration_doublePG,'-k',0:steps,Process_iteration_doublePG_extended,'--g',0:steps,Process_iteration_PG2d_extended,'-.b','LineWidth',4);
    title(pname,'FontSize',18);
    set(gca,'FontSize',16);
    saveas(gcf,pname,'epsc');
    %legend('PG_{2start} {+ constant stepsizes}','PG_{2start} {+ extended stepsizes}','PG_{2d} {+ extended stepsizes}','Location','southoutside','orientation','horizontal');

    decup(p);% destroy the CUTEst problem
end