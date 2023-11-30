clear; close all; clc

papersize = [3.5 3];
margins = [0 0];

paperposition = [margins, papersize(1)-2*margins(1), papersize(2)-2*margins(2)];

% define inverse problem
d = 6;
testcase = 'id-overdet';
problem = prob_setup(d,testcase);
problem.x = 1:d;
%%
% draw and visualize initial ensemble
J = 5;
V0 = problem.sample(J);
ensemblevis(problem,V0,'LS',1,'darkslides')
set(gcf,'papersize',papersize)
set(gcf,'paperposition',paperposition)
saveas(gcf,['figs/bdayod4_1.pdf'])


%% EKI iterations
num_iter = 50;
Vd = zeros(d,J,num_iter);

% identity dynamics version
Vd(:,:,1) = V0;
for i = 2:num_iter
    Vd(:,:,i) = EKIupdate(squeeze(Vd(:,:,i-1)),problem,'a','dzh');
    if i <=10 | mod(i,10)==0
        ensemblevis(problem,squeeze(Vd(:,:,i)),'LS',i,'darkslides')
        set(gcf,'papersize',papersize)
        set(gcf,'paperposition',paperposition)
        saveas(gcf,['figs/bdayod4_',num2str(i),'.pdf'])
    end
end

[Q,~] = qr(problem.G',0);
Pi = Q*Q';

covnorm = zeros(num_iter,3);
for i = 1:num_iter
    Vnow = squeeze(Vd(:,:,i));
    mu_i = mean(Vnow,2);
    Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
    covnorm(i,1) = norm(Gam_i);
    PGam_i  = Pi*Gam_i*Pi';
    Qi = eye(problem.d)-Pi;
    covnorm(i,2) = norm(PGam_i);
    covnorm(i,3) = norm(Gam_i - PGam_i);
    covnorm(i,4) = norm(Qi*Gam_i*Qi);
end

%%
% figure(1000); clf
% semilogy(covnorm(:,2:3)); hold on
% plot(1:num_iter, 0.005./(1:num_iter),'k:')
% xlabel('EKI iteration \#')
% legend({'$\|\hat\Gamma_i\|$','$\|\Gamma_i-\hat\Gamma_i\|$','$\frac1{2i}$ rate'},'interpreter','latex')