close all; clear; clc

recur_delta2_9a = @(d) d./((1+d).^2);  % (9a) case
recur_delta2_9c = @(d2) d2./(1+d2);    % (9c) case with idealized noise covariance

n = 5;
n_iter = 100;

delta2_9a = zeros(n,n_iter);
delta2_9c = zeros(n,n_iter);
ai_9c     = zeros(n,n_iter);
delta2_9a(:,1) = 10*rand(n,1);
delta2_9c(:,1) = delta2_9a(:,1);

for i = 2:n_iter
    delta2_9a(:,i) = recur_delta2_9a(delta2_9a(:,i-1));
    delta2_9c(:,i) = recur_delta2_9c(delta2_9c(:,i-1));
    ai_9c(:,i) = 1-(i-1)*delta2_9c(:,i);
end

%%
figure(1); clf; 
for i = 1:n
    loglog(delta2_9a(i,:),'Color',[0 ((i-1)*0.05+0.6) ((i-1)*0.2+0.2)]);
    hold on
    loglog(delta2_9c(i,:),':','Color',[0 ((i-1)*0.05+0.6) ((i-1)*0.2+0.2)]);
end

loglog(1./(2*(1:n_iter)),'k')
loglog(1./((1:n_iter)-1),'k:')

figure(2); clf
p = 1:1000;
semilogy(log(1+1./p),'k'); hold on
semilogy(1./(p+1),'b:')
semilogy(1./p,'r:')