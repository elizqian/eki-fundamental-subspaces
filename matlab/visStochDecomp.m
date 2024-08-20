clear; close all; clc

addpath('src/')

% define inverse problem
H     = [1 0 0; 0 1 0; 0 0 0];
[n,d] = size(H);
LSig = 0.49*rand(n,n);
Sigma = LSig*LSig';
truth = rand(n,1);
m     = H*truth+mvnrnd(zeros(1,n),Sigma)';
fisher = H'*(Sigma\H);
Hplus = pinv(fisher)*(H/Sigma);
vstar = Hplus*m;
iter  = 0;

problem = struct();
problem = add2struct(problem,H,n,d,Sigma,truth,m,fisher,Hplus,vstar,iter);

% initialize particles
J   = 15;
vv0 = [rand(1,J); zeros(1,J); 1:J];
% EKI iterations
max_iter  = 1000;
[vv,vvtilde]   = deal(zeros(d,J,max_iter+1));
noise = zeros(d,J,max_iter);
[vv(:,:,1),vvtilde(:,:,1)] = deal(vv0);

problem.Gi = cov(vv0');
for i = 1:max_iter
    vv(:,:,i+1) = EKIupdate(vv(:,:,i),problem,'stoch','iglesias');

    [vvtilde(:,:,i+1),problem] = EKIupdate(vvtilde(:,:,i),problem,'stoch','stoch-simple');
    noise(:,:,i) = problem.noise;
end

% define projections
spdc = specdecomp(H,vv0,Sigma);

% plot components of true stochastic iteration and tilde iteration
plotComponents(vv,problem,spdc,1)
sgtitle("stochastic EKI: misfit/error components/projections")

plotComponents(vvtilde,problem,spdc,2)
sgtitle("stochastic EKI: tilde iteration components")

% divide Pr components in deterministic and stochastic parts
deltas = reshape(   (1/spdc.delta(1) + (1:max_iter)).^-1,    1,1,max_iter);
hhtilde    = pagemtimes(H,vvtilde);
thetatilde = hhtilde-m;
omegatilde = vvtilde-vstar;
PrThTil = pagemtimes(spdc.calPr,thetatilde(:,:,2:end));
PrTh0 = spdc.calPr*thetatilde(:,:,1);
det = pagemtimes(deltas/spdc.delta(1),PrTh0);
rnd = pagemtimes(pagemtimes(deltas,spdc.calPr),cumsum(noise,3));
% hard coded for 3D example
total = squeeze(PrThTil(1,:,:));
det   = squeeze(det(1,:,:));
rnd   = squeeze(rnd(1,:,:));

%%
figure(3); clf
tot = loglog(abs(total'),'Color',[73 160 242 100]/255); hold on
dt = loglog(abs(det'),'Color',[169, 141, 240, 100]/255);
lin = loglog(1:max_iter, max(total,[],'all')./(1:max_iter),'k');
half = loglog(1:max_iter, max(total,[],'all')*(1:max_iter).^-0.5,'k:');
xlabel('$i$','interpreter','latex','fontsize',16)

legend([lin,half,tot(1),dt(1)],{'$1/i$','$1/\sqrt{i}$',...
    '$\mathcal{P}_r\tilde{\theta}_i$','$\mathcal{P}_r\tilde{\theta}_i$ deterministic part only'},'interpreter','latex',...
    'location','southwest','fontsize',18); legend boxoff