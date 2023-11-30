clear; close all; clc

% define inverse problem
d = 6;
testcase = 'id-overdet';
problem = prob_setup(d,testcase);

% draw and visualize initial ensemble
J = 21;
V0 = problem.sample(J);
ensemblevis(problem,V0,'LS',1)

%% EKI iterations
num_iter = 500;
Vd = zeros(d,J,num_iter);

% identity dynamics version
Vd(:,:,1) = V0;
for i = 2:num_iter
    Vd(:,:,i) = EKIupdate(squeeze(Vd(:,:,i-1)),problem,'c','dzh');
    if i <=10 | mod(i,100)==0
        ensemblevis(problem,squeeze(Vd(:,:,i)),'LS',i)
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
%     covnorm(i,3) = norm(Gam_i - PGam_i);
    covnorm(i,3) = norm(Pi*Gam_i*Qi');
    covnorm(i,4) = norm(Qi*Gam_i*Qi);
end

%%
figure(1000); clf
semilogy(covnorm(:,2:4)); hold on
plot(1:num_iter, 0.005./(1:num_iter),'k:')
xlabel('EKI iteration \#')
legend({'$\|\Pi\Gamma_i\Pi^T\|$','$\|\Pi\Gamma_i(I-\Pi)^T\|$','$\|(I-\Pi)\Gamma_i(I-\Pi)^T\|$','$\frac1{2i}$ rate'},'interpreter','latex','fontsize',18);
legend boxoff