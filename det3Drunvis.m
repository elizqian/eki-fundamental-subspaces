clear; close all; clc

addpath('src/')

% define inverse problem
H     = [1 0 0; 0 1 0; 0 0 0];
% H = eye(3);
[n,d] = size(H);
LSig = 0.49*rand(n,n);
Sigma = LSig*LSig';
truth = rand(n,1);
m     = H*truth+mvnrnd(zeros(1,n),Sigma)';
Hplus = pinv(H'*(Sigma\H))*(H/Sigma);
vstar = Hplus*m;
iter  = 0;

problem = struct();
problem = add2struct(problem,H,n,d,Sigma,truth,m,Hplus,vstar,iter);

% initialize particles
J   = 15;
vv0 = [rand(1,J); zeros(1,J); 1:J];
% EKI iterations
max_iter  = 1000;
vv        = zeros(d,J,max_iter+1);
vv(:,:,1) = vv0;
for i = 1:max_iter
    vv(:,:,i+1) = EKIupdate(vv(:,:,i),problem,'det','iglesias');
end

% define projections
spdc = specdecomp(H,vv0,Sigma);


% post-process
hh    = pagemtimes(H,vv);
theta = hh-m;
omega = vv-vstar;

%%
meas_projs = {'calPr','calQr','calNr'};
state_projs = {'bbPr','bbQr','bbNr'};
labels = {'$\theta_1$','$\mathcal{P}_r\theta$','$\omega_1$','$P_r\omega$',...
          '$\theta_2$','$\mathcal{Q}_r\theta$','$\omega_2$','$Q_r\omega$',...
          '$\theta_3$','$\mathcal{N}_r\theta$','$\omega_3$','$N_r\omega$'};
figure(1); clf
for i = 1:3
    subplot(3,4,(i-1)*4+1)
    loglog(0:max_iter,abs(squeeze(theta(i,:,:))'),':','Color',[73 160 242 150]/255)
    title(labels{(i-1)*4+1},'interpreter','latex','fontsize',20)

    subplot(3,4,(i-1)*4+2)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(meas_projs{i}),theta).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[73 160 242 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+2},'interpreter','latex','fontsize',20)

    subplot(3,4,(i-1)*4+3)
    loglog(0:max_iter,abs(squeeze(omega(i,:,:))'),':','Color',[245 187 42 150]/255)
    title(labels{(i-1)*4+3},'interpreter','latex','fontsize',20)

    subplot(3,4,(i-1)*4+4)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(state_projs{i}),omega).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[245 187 42 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+4},'interpreter','latex','fontsize',20)
end

sgtitle("deterministic EKI: misfit/error components/projections")