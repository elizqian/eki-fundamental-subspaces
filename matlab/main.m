clear; close all; clc

addpath('src/')

%% Problem set-up
% dimensions
n = 8;
d = 12;
J = 40;

% define random H, Sigma, and ensemble so all subspaces are non-trivial
[H,Sigma,v0] = EKIsetupRandom(n,d,J);

% generate measurement and least squares solution
truth = rand(d,1);
m     = H*truth+mvnrnd(zeros(1,n),Sigma)';
Hplus = pinv(H'*(Sigma\H))*(H'/Sigma);
vstar = Hplus*m;

% set up problem struct for EKIupdate function
problem = struct();
problem = add2struct(problem,H,n,d,Sigma,truth,m,Hplus,vstar);

%% run EKI experiments (deterministic, stochastic large and small ensembles)
Jsmall      = 5;        % size of small ensemble for deterministic and stochastic small ensemble
max_iter    = 1000;     % maximum iterations

% pre allocate and initialize ensembles
vSLarge                     = zeros(d,J,max_iter+1);
vSLarge(:,:,1)              = v0;
[vSSmall,vD]                = deal(zeros(d,Jsmall,max_iter+1));
[vD(:,:,1),vSSmall(:,:,1)]  = deal(v0(:,1:Jsmall));

% iterate
for i = 1:max_iter
    vD(:,:,i+1)      = EKIupdate(vD(:,:,i), problem, 'deterministic','adjoint-free');
    vSSmall(:,:,i+1) = EKIupdate(vSSmall(:,:,i),problem,'stochastic','adjoint-free');
    vSLarge(:,:,i+1) = EKIupdate(vSLarge(:,:,i),problem,'stochastic','adjoint-free');
end

% define projections
spdcSmall = specdecomp(H,v0(:,1:Jsmall),Sigma);
spdcLarge = specdecomp(H,v0,Sigma);

% post-process
hh    = pagemtimes(H,vD);
theta = hh-m;
omega = vD-vstar;

%% plot convergence
orange = [0.9137254901960784, 0.44313725490196076, 0.19607843137254902, 0.3];
blue   = [0.41568627450980394, 0.7372549019607844, 0.9215686274509803, 0.3];
gray   = [0.5, 0.5, 0.5, 0.3];

EKIs = {vD,vSLarge,vSSmall};
spdcs = {spdcSmall,spdcLarge,spdcSmall};
colors = {blue, orange ,gray};
sty = {'-','--',':'};
obs_projs = {'calP','calQ','calN'};
state_projs = {'bbP','bbQ','bbN'};
lbls = {'Deterministic $J = 5$','Stochastic $J = 15$','Stochastic $J = 5$'};
figure(1); clf
for i = 1:3     % loop through EKI experiments (columns)
    hh = pagemtimes(H,EKIs{i});
    theta = hh-m;
    omega = EKIs{i}-vstar;
    spdc  = spdcs{i};

    % top plot (observation space)
    subplot(2,3,i)
    lines = [];
    for j = 1:3 % loop through subspaces
        temp = loglog(0:max_iter,getComponentNorm(theta,spdc,obs_projs{j}),sty{j},'Color',colors{j}); hold on
        lines = [lines, temp(1)];
    end
    legend(lines,{'$\|\mathbf{\mathcal{P}}\mathbf{\theta}_i^{(j)}\|$',...
        '$\|\mathbf{\mathcal{Q}}\mathbf{\theta}_i^{(j)}\|$',...
        '$\|\mathbf{\mathcal{N}}\mathbf{\theta}_i^{(j)}\|$'},...
        'Location','Best', 'interpreter','latex'); legend boxoff
    title(lbls{i}, 'interpreter','latex')

    % bottom plot (state space)
    subplot(2,3,i+3)
    lines = [];
    for j = 1:3 % loop through subspaces
        temp = loglog(0:max_iter,getComponentNorm(omega,spdc,state_projs{j}),sty{j},'Color',colors{j}); hold on
        lines = [lines, temp(1)];
    end
    legend(lines,{'$\|P\mathbf{\omega}_i^{(j)}\|$',...
        '$\|Q\mathbf{\omega}_i^{(j)}\|$',...
        '$\|N\mathbf{\omega}_i^{(j)}\|$'},...
        'Location','Best', 'interpreter','latex'); legend boxoff

end
subplot(2,3,1); ylabel('Observation space','interpreter','latex')
subplot(2,3,4); ylabel('State space','interpreter','latex')
subplot(2,3,5); xlabel('Iteration number $i$','interpreter','latex')

function [compNorm] = getComponentNorm(qoi,spdc,comp)
projected_qoi = pagemtimes(spdc.(comp),qoi);
compNorm = sqrt(squeeze(sum(projected_qoi.^2,1)))';
end