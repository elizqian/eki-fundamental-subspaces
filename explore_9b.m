clear; close all; clc

% define inverse problem
d = 6;
testcase = 'id-sparse';
problem = prob_setup(d,testcase);

% draw and visualize initial ensemble
J = 10001;
V0 = problem.sample(J);
ensemblevis(problem,V0,'LS',1)

%% EKI iterations
num_iter = 10;
Vd = zeros(d,J,num_iter);

gevs = zeros(problem.n,num_iter);

% identity dynamics version
Vd(:,:,1) = V0;
mu0 = mean(V0,2);
Gam_i = (V0-mu0)*(V0-mu0)'/(J-1);
[V,D] = eig(problem.G*Gam_i*problem.G', problem.Geps);
temp = repmat(V(1,:) < 0,problem.n,1);
V(temp) = -V(temp);
[B,I] = sort(V(1,:));
V = V(:,I);
D = diag(D);
D = D(I);

gevs(:,1) = D;

for i = 2:num_iter
    [Vnow, problem] = EKIupdate(squeeze(Vd(:,:,i-1)),problem,'b','dzh');
    Vd(:,:,i) = Vnow;
    if i <=10 | mod(i,100)==0
        ensemblevis(problem,squeeze(Vd(:,:,i)),'LS',i)
    end

    mu_now = mean(Vnow,2);
    Gam_i = (Vnow-mu_now)*(Vnow-mu_now)'/(J-1);
    [V,D] = eig(problem.G*Gam_i*problem.G', problem.Geps);
    temp = repmat(V(1,:) < 0,problem.n,1);
    V(temp) = -V(temp);
    [B,I] = sort(V(1,:));
    V = V(:,I);
    D = diag(D);
    D = D(I);
    gevs(:,i) = D;
end

%%
figure(11); clf
plot(gevs(:,1)); hold on
plot(gevs(:,2))