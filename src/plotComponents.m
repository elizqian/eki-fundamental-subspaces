function plotComponents(vv,problem,spdc,fignum)

H = problem.H;
m = problem.m;
vstar = problem.vstar;

max_iter = size(vv,3)-1;

hh    = pagemtimes(H,vv);
theta = hh-m;
omega = vv-vstar;

meas_projs = {'calPr','calQr','calNr'};
state_projs = {'bbPr','bbQr','bbNr'};
labels = {'$\theta_1$','$\mathcal{P}_r\theta$','$\omega_1$','$P_r\omega$',...
          '$\theta_2$','$\mathcal{Q}_r\theta$','$\omega_2$','$Q_r\omega$',...
          '$\theta_3$','$\mathcal{N}_r\theta$','$\omega_3$','$N_r\omega$'};
figure(fignum); clf
for i = 1:3
    subplot(3,4,(i-1)*4+1)
    loglog(0:max_iter,abs(squeeze(theta(i,:,:))'),':','Color',[73 160 242 150]/255)
    title(labels{(i-1)*4+1},'interpreter','latex')

    subplot(3,4,(i-1)*4+2)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(meas_projs{i}),theta).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[73 160 242 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+2},'interpreter','latex')

    subplot(3,4,(i-1)*4+3)
    loglog(0:max_iter,abs(squeeze(omega(i,:,:))'),':','Color',[245 187 42 150]/255)
    title(labels{(i-1)*4+3},'interpreter','latex')

    subplot(3,4,(i-1)*4+4)
    derp = sqrt(squeeze(sum(pagemtimes(spdc.(state_projs{i}),omega).^2,1))');
    if i == 1
        loglog(1:max_iter,max(derp(2,:),[],'all')*1./sqrt(1:max_iter),'Color',[0.5 0.5 0.5]); hold on
    end
    loglog(0:max_iter,derp,'--','Color',[245 187 42 150]/255)
    ytickformat('%.2f')
    title(labels{(i-1)*4+4},'interpreter','latex')
end
