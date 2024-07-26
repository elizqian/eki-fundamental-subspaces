clear; close all; clc

recur_delta_vanilla = @(delta) delta./ ((1+delta).^2);
recur_delta_accel   = @(lam,varpi) lam.*((1 + lam - varpi*lam)./(1+lam)).^2;

% desired convergence rate in the accelerated version
p = 10;

max_iter = 10000;
delt0 = logspace(-4,-0.001,4);
[deltsAccel,deltsVanil] = deal(zeros(length(delt0),max_iter));
[deltsVanil(:,1),indD] = sort(recur_delta_vanilla(delt0),'descend');
[deltsAccel(:,1),indL] = sort(recur_delta_accel(delt0,2),'descend');
for i = 2:max_iter
    deltsVanil(:,i) = recur_delta_vanilla(deltsVanil(:,i-1));

    % see where p enters into the definition of varpi
    varpi = 2*i^(p-1);
    varpi = exp(i);
    deltsAccel(:,i) = recur_delta_accel(deltsAccel(:,i-1),varpi);
end

%%
figure(1); clf
subplot(1,2,1)
loglog(1:max_iter,deltsVanil'); hold on
h1 = loglog(1:max_iter,1./(1:max_iter),'k');
h2 = loglog(1:max_iter,1./((1:max_iter).^(p)),'k:');
ylim([1e-9 1e0])
legend([h1, h2],{'$1/i$',['$1/i^',num2str(p),'$']},'interpreter','latex','location','northeast','fontsize',14); legend boxoff
xlabel('Iteration number $i$','interpreter','latex')
ylabel('$\delta_{\ell,i}$','interpreter','latex')
title('Unaccelerated deterministic eigenvalues')

subplot(1,2,2)
loglog(1:max_iter,deltsAccel'); hold on 
h1 = loglog(1:max_iter,1./(1:max_iter),'k');
h2 = loglog(1:max_iter,1./((1:max_iter).^(p)),'k:');
% ylim([1e-9 1e0])
xlabel('Iteration number $i$','interpreter','latex')
ylabel('$\lambda_{\ell,i}$','interpreter','latex')
title('Accelerated deterministic eigenvalues')
