clear; close all; clc

% papersize = [3.5 3];
% margins = [0 0];
% 
% paperposition = [margins, papersize(1)-2*margins(1), papersize(2)-2*margins(2)];

% define inverse problem
d = 6;
testcase = 'id-test';
problem = prob_setup(d,testcase);
problem.x = 1:d;
%%
% draw and visualize initial ensemble
J = 5;


V0 = [0 1 3 0 0 0; ...
      0 2 0 0 0 0; ...
      0 0 1 0 0 0; ... 
      0 0 1 1 0 0; ...
      0 1 0 0 1 0]';

h = rank(problem.fisher);
gm = rank(cov(V0'));

[sig2,U,M,Vhat,Lam,What] = solve_state_eig(V0,problem);

[Phi,Sig,Psi] = svd(Vhat(:,1:gm)'*What(:,1:h));
r = rank(Sig);
thing = Vhat(:,1:gm)*M(1:gm,1:gm)*Phi(:,1:r);


ensemblevis(problem,V0,'LS',1,'darkslides')
% set(gcf,'papersize',papersize)
% set(gcf,'paperposition',paperposition)
% saveas(gcf,['figs/bdayod4_1.pdf'])


%% EKI iterations
num_iter = 100;
Vd = zeros(d,J,num_iter);
coeff = zeros(d,J,num_iter);
coeff2 = zeros(d,J,num_iter);

Vd(:,:,1) = V0;
coeff(:,:,1) = U'*V0;
coeff2(:,:,1) = U'*fisher*V0;
for i = 2:num_iter
    Vd(:,:,i) = EKIupdate(squeeze(Vd(:,:,i-1)),problem,'a','dzh');
    coeff(:,:,i) = U'*squeeze(Vd(:,:,i));
    coeff2(:,:,i) = U'*fisher*squeeze(Vd(:,:,i));
    if i <=10 | mod(i,10)==0
        ensemblevis(problem,squeeze(Vd(:,:,i)),'LS',i,'darkslides')
%         set(gcf,'papersize',papersize)
%         set(gcf,'paperposition',paperposition)
%         saveas(gcf,['figs/bdayod4_',num2str(i),'.pdf'])
    end
end

%%
figure(101); clf
for i = 1:6
subplot(2,6,i)
plot(squeeze(coeff(i,:,:))')
subplot(2,6,i+6)
loglog(abs(squeeze(coeff(i,:,:)-coeff(i,:,end))'))
end

figure(102); clf
for i = 1:6
subplot(2,6,i)
plot(squeeze(coeff2(i,:,:))')
subplot(2,6,i+6)
loglog(abs(squeeze(coeff2(i,:,:)-coeff2(i,:,end))'))
end

[Q,~] = qr(problem.G',0);
Pi = Q*Q';


covnorm = zeros(num_iter,3);
for i = 1:num_iter
    Vnow = squeeze(Vd(:,:,i));
    mu_i = mean(Vnow,2);
    Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
    covnorm(i,1) = norm(Gam_i);
    PGam_i  = Pi*Gam_i*Pi';
    Qi = eye(problem.d)-Pi;
    covnorm(i,2) = norm(PGam_i);
    covnorm(i,3) = norm(Gam_i - PGam_i);
    covnorm(i,4) = norm(Qi*Gam_i*Qi);
end
