clear; close all; clc

addpath('src/')

% define inverse problem
H     = [1 0 0; 0 1 0; 0 0 0];
[n,d] = size(H);
LSig = 0.49*rand(n,n);
Sigma = LSig*LSig';
truth = rand(n,1);
m     = H*truth+mvnrnd(zeros(1,n),Sigma)';
Hplus = pinv(H'*(Sigma\H))*(H/Sigma);
vstar = Hplus*m;
iter  = 0;

problemDefault = struct();
problemDefault = add2struct(problemDefault,H,n,d,Sigma,truth,m,Hplus,vstar,iter);
problemRichardson = problemDefault;

% initialize particles
J   = 15;
vv0 = [rand(1,J); zeros(1,J); 1:J];
% EKI iterations
max_iter  = 1000;
[vvD, vvR] = deal(zeros(d,J,max_iter+1));
vvD(:,:,1) = vv0;
vvR(:,:,1) = vv0;
for i = 1:max_iter
    vvD(:,:,i+1) = EKIupdate(vvD(:,:,i),problemDefault,'det','iglesias');
    [temp, problemRichardson] = EKIupdate(vvR(:,:,i),problemRichardson,'det','richardson');
    vvR(:,:,i+1) = temp;
end

% define projections
spdc = specdecomp(H,vv0,Sigma);

% post-process
hhD    = pagemtimes(H,vvD);
thetaD = hhD-m;
omegaD = vvD-vstar;

hhR    = pagemtimes(H,vvR);
thetaR = hhR-m;
omegaR = vvR-vstar;

%%
meas_projs = {'calPr','calQr','calNr'};
state_projs = {'bbPr','bbQr','bbNr'};
labels = {'$\theta_1$','$\mathcal{P}_r\theta$','$\omega_1$','$P_r\omega$',...
          '$\theta_2$','$\mathcal{Q}_r\theta$','$\omega_2$','$Q_r\omega$',...
          '$\theta_3$','$\mathcal{N}_r\theta$','$\omega_3$','$N_r\omega$'};
figure(1); clf
for i = 1:3
    subplot(3,4,(i-1)*4+1)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(meas_projs{i}),thetaD).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[73 160 242 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+2},'interpreter','latex','fontsize',20)

    subplot(3,4,(i-1)*4+2)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(meas_projs{i}),thetaR).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[245 187 42 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+2},'interpreter','latex','fontsize',20)

    subplot(3,4,(i-1)*4+3)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(state_projs{i}),omegaD).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[73 160 242 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+4},'interpreter','latex','fontsize',20)

    subplot(3,4,(i-1)*4+4)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(state_projs{i}),omegaR).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[245 187 42 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+4},'interpreter','latex','fontsize',20)
end

sgtitle("left/blue: default, right/gold: with accel param")