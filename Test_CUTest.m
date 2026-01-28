clc;
clear;
problist = secup(struct('type', 'u', 'mindim', 100, 'maxdim', 2000));
%problist ={'BROWNBS','FLETCBV3','HYDC20LS'};
%problist ={'COATING'};
%problist ={'PENALTY2'};
%problist ={'EDENSCH'};
%problist ={'PENALTY3'};
%problist ={'SPINLS'};
%problist ={'ARGLINC'};
%problist ={'ARGLINB'};
%problist ={'FLETCHCR'};
%unbounded_list={'10FOLDTRLS','FLETCBV3','FLETCHBV','INDEF'}
a=clock;
filename=['TRS' '--' date '-' int2str(a(4)) '-' int2str(a(5)) '.txt'];
fid = fopen(filename,'w');
fprintf(fid,'Problem & Method & k & k_{prods} & k_f & k_{gradshess} & norm(g) & f & time\n');
%%TR method setting
epsilon_g = 1e-5;
eta_1 = 0.01;
eta_2 = 0.95;
gamma_1 = 0.5;
gamma_2 = 2;
K = 1e4; %iterative step limit uper bound
Delta = 4;
epsilon_Delta = 1e-10;
%for ip = 14: 14
for ip = 1: length(problist)
    pname = problist{ip};
    p = macup(pname);  % make a CUTEst problem
    
    x0=p.x0;
    dimx = length(p.x0);
    fprintf('\n%d. Solve %s:\n', ip, pname);

    
    %%%data collection, data in each time

    repeats = 1;
    Data_C=zeros(repeats,7);
    Data_ST=zeros(repeats,7);
    Data_ST_pre=zeros(repeats,7);
    Data_P=zeros(repeats,7);
    Data_GEP=zeros(repeats,7);
%     Data_Pa=zeros(repeats,7);
    for ii=1:repeats
        fprintf('------------------START of TR_Cauchy------------------\n');
        tstart_Cauchy=tic;
        [~,f_C,norm_g_C,step_C,k_f_C,k_gradshess_C,~] = TR_Cauchy(p,dimx,x0,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta);
        time_Cauchy=toc(tstart_Cauchy);
        fprintf('------------------END of TR_Cauchy------------------\n');
        
        fprintf('------------------START of TR_ST------------------\n');
        tstart_ST=tic;
        [~,f_ST,norm_g_ST,step_ST,k_f_ST,k_gradshess_ST,k_prods_ST] = TR_ST(p,dimx,x0,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta);
        time_ST=toc(tstart_ST);
        fprintf('------------------END of TR_ST------------------\n');
        
        fprintf('------------------START of TR_ST_precondition------------------\n');
        tstart_ST_pre=tic;
        [~,f_ST_pre,norm_g_ST_pre,step_ST_pre,k_f_ST_pre,k_gradshess_ST_pre,k_prods_ST_pre] = TR_ST_precondition(p,dimx,x0,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta);
        time_ST_pre=toc(tstart_ST_pre);
        fprintf('------------------END of TR_ST_precondition------------------\n'); 
%       fprintf('------------------START of TR_GEP------------------\n');
%       tstart_GEP=tic;
%       [~,f_GEP,norm_g_GEP,step_GEP,k_f_GEP,k_gradshess_GEP,H_GEP] = TR_GEP(p,dimx,x0,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta);
%        time_GEP=toc(tstart_GEP);
%        fprintf('------------------END of TR_GEP------------------\n'); 
        fprintf('------------------START of TR_PG------------------\n');
        tstart_P=tic;
        [~,f_P,norm_g_P,step_P,k_f_P,k_gradshess_P,H_P] = TR_PG(p,dimx,x0,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta,epsilon_Delta);
        time_P=toc(tstart_P);
        fprintf('------------------END of TR_PG------------------\n');
%         
%         fprintf('------------------START of TR_PG_highacc------------------\n');
%         tstart_Pa=tic;
%         [~,f_Pa,norm_g_Pa,step_Pa,k_f_Pa,k_gradshess_Pa] = TR_PG_highacc(p,dimx,x0,epsilon_g,eta_1,eta_2,gamma_1,gamma_2,K,Delta);
%         time_Pa=toc(tstart_Pa);
%         fprintf('------------------END of TR_PG_highacc------------------\n');
        k_prods_C=step_C;
        k_prods_P=sum(ceil((1:step_P)/5));
        temp=eig(H_P);
        con_number=min(temp)/max(temp);
        %k_prods_Pa=sum(1:step_Pa);
        Data_C(ii,:)=[step_C,k_prods_C,k_f_C,k_gradshess_C,norm_g_C,f_C,time_Cauchy];
        Data_ST(ii,:)=[step_ST,k_prods_ST,k_f_ST,k_gradshess_ST,norm_g_ST,f_ST,time_ST];
        Data_ST_pre(ii,:)=[step_ST_pre,k_prods_ST_pre,k_f_ST_pre,k_gradshess_ST_pre,norm_g_ST_pre,f_ST_pre,time_ST_pre];
        Data_P(ii,:)=[step_P,k_prods_P,k_f_P,k_gradshess_P,norm_g_P,f_P,time_P];
%         Data_Pa(ii,:)=[step_Pa,k_prods_Pa,k_f_Pa,k_gradshess_Pa,norm_g_Pa,f_Pa,time_Pa];
%         Data_GEP(ii,:)=[step_GEP,k_f_GEP,k_f_GEP,k_gradshess_GEP,norm_g_GEP,f_GEP,time_GEP];
                
    end
    fprintf(fid,'pname = %6s, dimx = %4g, con_number = %1.2e \n', pname, dimx, con_number);
    fprintf(fid,'%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_Cauchy', sum(Data_C(1:repeats,:),1)/repeats);
    fprintf(fid,'%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_ST', sum(Data_ST(1:repeats,:),1)/repeats);
    fprintf(fid,'%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_ST_pre', sum(Data_ST_pre(1:repeats,:),1)/repeats);
    fprintf(fid,'%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_PG',  sum(Data_P(1:repeats,:),1)/repeats);
    %%%%
    fprintf('Problem & Method & k & k_{prods} & k_f & k_{gradshess} & norm(g) & f & time\n');
    fprintf('pname = %6s, dimx = %4g, con_number = %1.2e \n', pname, dimx, con_number);
    fprintf('%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_Cauchy', sum(Data_C(1:repeats,:),1)/repeats);
    fprintf('%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_ST', sum(Data_ST(1:repeats,:),1)/repeats);
    fprintf('%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_ST_pre', sum(Data_ST_pre(1:repeats,:),1)/repeats);
    fprintf('%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_PG',  sum(Data_P(1:repeats,:),1)/repeats);
%     fprintf(fid,'%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_PG2',sum(Data_Pa(1:repeats,:),1)/repeats);
%    fprintf(fid,'%6s(%4g) & %10s & %6.1f & %6.1f & %6.1f & %6.1f & %1.2e & %1.2e & %5.3f\n', pname, dimx, 'TRS_GEP',sum(Data_GEP(1:repeats,:),1)/repeats);
decup(p);% destroy the CUTEst problem
end
fclose(fid);