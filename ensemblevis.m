function ensemblevis(problem,samples,fignum)

if nargin == 2
    figure; 
else
    figure(fignum); clf
end
subplot(2,1,1)
s0 = plot(problem.x,samples,'Color',[0 0.447 0.741 0.1]); hold on
s1 = plot(problem.x,problem.truth,'k'); hold on
s2 = plot(problem.x,problem.mupos,'r:');
legend([s0(1),s1,s2],{'$u\sim\pi_{\rm pr}$','$u^\dagger$','$\mu_{\rm pos}$'},'interpreter','latex','location','southwest','orientation','horizontal'); 
title('State space $R^d$','interpreter','latex')
subplot(2,1,2)
m0 = plot(1:problem.n,problem.G*samples,'Color',[255 154 113 25.5]/255); hold on
m1 = plot(1:problem.n,problem.G*problem.truth,'k'); 
m2 = plot(1:problem.n, problem.meas,'+','Color',[0.3 0.3 0.3 0.5],'MarkerSize',2);
m3 = plot(1:problem.n,problem.G*problem.mupos,'r:');
xlim([1 problem.n])
legend([m0(1),m1,m2,m3],{'$Gu\sim G_\#\pi_{\rm pr}$','$Gu^\dagger$','$m$','$G\mu_{\rm pos}$'},...
    'interpreter','latex','location','south','orientation','horizontal'); 
title('Measurement space $R^n$')