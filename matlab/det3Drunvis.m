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
Hplus = pinv(H'*(Sigma\H))*(H'/Sigma);
vstar = Hplus*m;
iter  = 0;

problem = struct();
problem = add2struct(problem,H,n,d,Sigma,truth,m,Hplus,vstar,iter);

% initialize particles
J   = 15;
vv0 = [rand(1,J); zeros(1,J); 0.5*randn(1,J)];
vv0 = [rand(1,J); 0.2*ones(1,J); 0.5*randn(1,J)];
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
%%
figure(2); clf
subplot(1,2,1)
derp = sqrt(squeeze(sum(pagemtimes(spdc.calPr,theta).^2,1))');
l1 = loglog(0:max_iter,derp,'Color',[73 160 242 120]/255); hold on
derp = sqrt(squeeze(sum(pagemtimes(spdc.calQr,theta).^2,1))');
l2 = loglog(0:max_iter,derp,'--','Color',[73 160 242 120]/255);
derp = sqrt(squeeze(sum(pagemtimes(spdc.calNr,theta).^2,1))');
l3 = loglog(0:max_iter,derp,':','Color',[73 160 242 120]/255);
legend([l1(1),l2(1),l3(1)],{'calPr','calQr','calNr'},'Location','Best')
xlabel('Iteration #')
title('Measurement space')

subplot(1,2,2)
derp = sqrt(squeeze(sum(pagemtimes(spdc.bbPr,omega).^2,1))');
l1 = loglog(0:max_iter,derp,'Color',[245 187 42 150]/255); hold on
derp = sqrt(squeeze(sum(pagemtimes(spdc.bbQr,omega).^2,1))');
l2 = loglog(0:max_iter,derp,'-.','Color',[245 187 42 150]/255);
derp = sqrt(squeeze(sum(pagemtimes(spdc.bbNr,omega).^2,1))');
l3 = loglog(0:max_iter,derp,':','Color',[245 187 42 150]/255);
legend([l1(1),l2(1),l3(1)],{'bbPr','bbQr','bbNr'},'Location','Best')
xlabel('Iteration #')
title('State space')

%% 
figure(3); clf
for j = 1:J
x = squeeze(vv(1,j,1:10:end));
y = squeeze(vv(3,j,1:10:end));
plot(x,y,'k'); hold on
start = plot(x(1),y(1),'r+');
finis = plot(x(end),y(end),'bo');
end
legend([start(1),finis(1)],{'v_0','v_T'},'fontsize',18,'location','best')

%%
figure(4); clf

temp = pagemtimes(spdc.bbPr,vv);
temp = vv;
for j = 1:J
    x = squeeze(temp(1,j,1:10:end));
    y = squeeze(temp(2,j,1:10:end));
    z = squeeze(temp(3,j,1:10:end));
    plot3(x,y,z,'k'); hold on
    start = plot3(x(1),y(1),z(1),'r+');
    finis = plot3(x(end),y(end),z(end),'bo');
end
star = plot3(vstar(1),vstar(2),vstar(3),'g*');
Pstar = spdc.bbPr*vstar;
ppstar = plot3(Pstar(1),Pstar(2),Pstar(3),'m*');
plot3([vstar(1) Pstar(1)],[vstar(2) Pstar(2)],[vstar(3) Pstar(3)],'g:')

kerPr = null(spdc.bbPr);
kerP = plotPlane(kerPr(:,1),kerPr(:,2));
set(kerP, 'FaceColor', 'yellow', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

ranGam0 = orth(vv0);
ranPart = plotPlane(ranGam0(:,1),ranGam0(:,2));
set(ranPart, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

ranH = orth(H);
ranH = plotPlane(ranH(:,1),ranH(:,2));
set(ranH, 'FaceColor', 'magenta', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

xlabel('x')
ylabel('y')
zlabel('z')
legend([start(1),finis(1),star,kerP,ranPart,ranH],{'v_0','v_T','v^*','ker(Pr)','ran(Gam0)','ran(H)'},'fontsize',18,'location','best')
% axis equal

function plane = plotPlane(v1,v2)
[u, v] = meshgrid(-1:1, -1:2);

% Define the plane using the basis vectors
X = u * v1(1) + v * v2(1);
Y = u * v1(2) + v * v2(2);
Z = u * v1(3) + v * v2(3);

plane = surf(X, Y, Z);

end