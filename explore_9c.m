clear; close all; clc

% define inverse problem
d = 6;
testcase = 'id-overdet';
problem = prob_setup(d,testcase);

% draw and visualize initial ensemble
J = 4;
V0 = problem.sample(J);
ensemblevis(problem,V0,'LS',1)

%% EKI iterations
num_iter = 1000;
[Vc,Va] = deal(zeros(d,J,num_iter));

% identity dynamics version
Vc(:,:,1) = V0;
Va(:,:,1) = V0;
for i = 2:num_iter
    Va(:,:,i) = EKIupdate(squeeze(Va(:,:,i-1)),problem,'a','dzh');
    Vc(:,:,i) = EKIupdate(squeeze(Vc(:,:,i-1)),problem,'c','dzh');
    if i <=10 | mod(i,100)==0
        %uncomment for ensemble visualizations
%         ensemblevis(problem,squeeze(Vc(:,:,i)),'LS',i)
    end
end

%% compute projectors
[Q,~] = qr(problem.G',0);
Pi = Q*Q';
Gam_0 = cov(V0');
r = min([problem.n, problem.d,rank(Gam_0)]);

% measurement space
[W,D] = eig(problem.G*Gam_0*problem.G',problem.Geps);
[D,ind] = sort(diag(D),'descend');
W = W(:,ind);
for i = 1:problem.n
    W(:,i) = W(:,i)./(W(:,i)'*problem.Geps*W(:,i));
end
calPr = problem.Geps*W(:,1:r)*W(:,1:r)';
calQr = problem.Geps*W(:,r+1:end)*W(:,r+1:end)';

% state space
[U,D] = eig(Gam_0*problem.fisher);
[D,ind] = sort(diag(D),'descend');
U = U(:,ind);
for i = 1:rank(problem.G)
    U(:,i) = U(:,i)./(U(:,i)'*problem.fisher*U(:,i));
end
bbPr = U(:,1:r)*U(:,1:r)'*problem.fisher;
bbQr = eye(problem.d) - bbPr;

%% post process misfits and covariances for plotting

% misfits measurement space
misfit_c = pagemtimes(problem.G,Vc) - problem.meas;
Pmisfit_c = pagemtimes(calPr,misfit_c);
Qmisfit_c = pagemtimes(calQr,misfit_c);
normtheta_c = squeeze(sqrt(sum(misfit_c.^2,1)));
normPtheta_c = squeeze(sqrt(sum(Pmisfit_c.^2,1)));
normQtheta_c = squeeze(sqrt(sum(Qmisfit_c.^2,1)));
Fnormtheta_c = squeeze(sqrt(sum(normtheta_c.^2,1)));
FnormPtheta_c = squeeze(sqrt(sum(normPtheta_c.^2,1)));
FnormQtheta_c = squeeze(sqrt(sum(normQtheta_c.^2,1)));


misfit_a = pagemtimes(problem.G,Va) - problem.meas;
Pmisfit_a = pagemtimes(calPr,misfit_a);
Qmisfit_a = pagemtimes(calQr,misfit_a);
normtheta_a = squeeze(sqrt(sum(misfit_a.^2,1)));
normQtheta_a = squeeze(sqrt(sum(Qmisfit_a.^2,1)));
normPtheta_a = squeeze(sqrt(sum(Pmisfit_a.^2,1)));
Fnormtheta_a = squeeze(sqrt(sum(normtheta_a.^2,1)));
FnormPtheta_a = squeeze(sqrt(sum(normPtheta_a.^2,1)));
FnormQtheta_a = squeeze(sqrt(sum(normQtheta_a.^2,1)));

% misfits state space
err_c = Vc - problem.muLS;
Perr_c = pagemtimes(bbPr,err_c);
Qerr_c = pagemtimes(bbQr,err_c);
temp = squeeze(sqrt(sum(err_c.^2,1)));
FOmega_c = squeeze(sqrt(sum(temp.^2,1)));
temp = squeeze(sqrt(sum(Perr_c.^2,1)));
FPOmega_c = squeeze(sqrt(sum(temp.^2,1)));
temp = squeeze(sqrt(sum(Qerr_c.^2,1)));
FQOmega_c = squeeze(sqrt(sum(temp.^2,1)));

err_a = Va - problem.muLS;
Perr_a = pagemtimes(bbPr,err_a);
Qerr_a = pagemtimes(bbQr,err_a);
temp = squeeze(sqrt(sum(err_a.^2,1)));
FOmega_a = squeeze(sqrt(sum(temp.^2,1)));
temp = squeeze(sqrt(sum(Perr_a.^2,1)));
FPOmega_a = squeeze(sqrt(sum(temp.^2,1)));
temp = squeeze(sqrt(sum(Qerr_a.^2,1)));
FQOmega_a = squeeze(sqrt(sum(temp.^2,1)));


% covariances
[covnorm_c,covnorm_a,Hcov_a,Hcov_c] = deal(zeros(num_iter,3));
for i = 1:num_iter
    Vnow = squeeze(Vc(:,:,i));
    mu_i = mean(Vnow,2);
    Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
    covnorm_c(i,1) = norm(Gam_i);
    covnorm_c(i,2) = norm(bbPr*Gam_i*bbPr');
    covnorm_c(i,3) = norm(bbQr*Gam_i*bbQr');
    temp = problem.G*Gam_i*problem.G';
    Hcov_c(i,1) = norm(temp);
    Hcov_c(i,2) = norm(calPr*temp*calPr');
    Hcov_c(i,3) = norm(calQr*temp*calQr');

    Vnow = squeeze(Va(:,:,i));
    mu_i = mean(Vnow,2);
    Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
    covnorm_a(i,1) = norm(Gam_i);
    covnorm_a(i,2) = norm(bbPr*Gam_i*bbPr');
    covnorm_a(i,3) = norm(bbQr*Gam_i*bbQr');
    temp = problem.G*Gam_i*problem.G';
    Hcov_a(i,1) = norm(temp);
    Hcov_a(i,2) = norm(calPr*temp*calPr');
    Hcov_a(i,3) = norm(calQr*temp*calQr');
end

%% convergence plots
% measurement space plots: misfit
figure(5000); clf
subplot(2,2,1)
loglog(Fnormtheta_a); hold on
loglog(FnormPtheta_a,':')
loglog(FnormQtheta_a,':')
plot(1:num_iter, 1./sqrt(1:num_iter),'Color',[0.5 0.5 0.5])
temp = get(gca,'colororder');
loglog(Fnormtheta_c,'o','Color',temp(1,:));
loglog(FnormPtheta_c,'+','Color',temp(2,:));
loglog(FnormQtheta_c,'+','Color',temp(3,:));
xlabel('EKI iteration \#', 'interpreter','latex','fontsize',20)
legend({'$\|\Theta_i\|_F$','$\|\mathcal{P}_r\Theta_i\|_F$','$\|\mathcal{Q}_r\Theta_i\|_F$',...
    '$\frac{1}{\sqrt{i}}$ rate'},'Location','Best','interpreter','latex','fontsize',20,'location','best')
legend boxoff
title('Misfit','interpreter','latex','Fontsize',20)
sgtitle('Lines are 9a, Markers are 9c','interpreter','latex','Fontsize',20)

% measurement space plots: covariance
subplot(2,2,2)
loglog(Hcov_a(:,1)); hold on
loglog(Hcov_a(:,2),':'); 
loglog(Hcov_a(:,3),':'); 
plot(1:num_iter, 0.001./(1:num_iter),'Color',[0.5 0.5 0.5])
temp = get(gca,'colororder');
loglog(Hcov_c(:,1),'o','Color',temp(1,:));
loglog(Hcov_c(:,2),'+','Color',temp(2,:));
loglog(Hcov_c(:,3),'+','Color',temp(3,:));
xlabel('EKI iteration \#', 'interpreter','latex')
legend({'$\|\Gamma_i\|$','$\|P_r\Gamma_iP_r^T\|$','$\|Q_r\Gamma_iQ_r^T\|$','$\frac1{i}$ rate'},...
    'interpreter','latex','fontsize',18,'location','best');
legend boxoff
title('Measurement Cov','interpreter','latex','Fontsize',20)


% state space plots: error
subplot(2,2,3)
loglog(FOmega_a); hold on
loglog(FPOmega_a,':');
loglog(FQOmega_a,':');
plot(1:num_iter, 0.001./sqrt(1:num_iter),'Color',[0.5 0.5 0.5])
loglog(FOmega_c,'o','Color',temp(1,:));
loglog(FPOmega_c,'o','Color',temp(2,:));
loglog(FQOmega_c,'+','Color',temp(3,:));
legend({'$\|\Omega_i\|_F$','$\|P_r\Omega_i\|_F$','$\|Q_r\Omega_i\|_F$','$\frac{1}{\sqrt{i}}$ rate'},'Location','Best','interpreter','latex','fontsize',20)
legend boxoff
title('State error','interpreter','latex','Fontsize',20)
xlabel('EKI iteration \#', 'interpreter','latex','fontsize',20)


% state space plots: covariance
subplot(2,2,4)
loglog(covnorm_a(:,1)); hold on
loglog(covnorm_a(:,2),':'); 
loglog(covnorm_a(:,3),':'); 
plot(1:num_iter, 0.001./(1:num_iter),'Color',[0.5 0.5 0.5])
temp = get(gca,'colororder');
loglog(covnorm_c(:,1),'o','Color',temp(1,:));
loglog(covnorm_c(:,2),'+','Color',temp(2,:));
loglog(covnorm_c(:,3),'+','Color',temp(3,:));
xlabel('EKI iteration \#', 'interpreter','latex')
legend({'$\|\Gamma_i\|$','$\|P_r\Gamma_iP_r^T\|$','$\|Q_r\Gamma_iQ_r^T\|$','$\frac1{i}$ rate'},'interpreter','latex','fontsize',18);
legend boxoff
title('State Cov','interpreter','latex','Fontsize',20)