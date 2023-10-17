clear; close all; clc

% define inverse problem
d = 201;
testcase = 'id-sparse';
problem = prob_setup(d,testcase);

% draw and visualize initial ensemble
J = 100;
V0 = problem.sample(J);
ensemblevis(problem,V0,1)

%% EKI iterations
num_iter = 10;
[Vd,Vi] = deal(zeros(d,J,num_iter));

% identity dynamics version
Vd(:,:,1) = V0;
for i = 2:num_iter
    Vd(:,:,i) = EKIupdate(squeeze(Vd(:,:,i-1)),problem,'a','dzh');
    ensemblevis(problem,squeeze(Vd(:,:,i)),i)
end

% Iglesias formulation
Vi(:,:,1) = V0;
for i = 2:num_iter
    Vi(:,:,i) = EKIupdate(squeeze(Vi(:,:,i-1)),problem,'a','iglesias');
    ensemblevis(problem,squeeze(Vi(:,:,i)),100+i)
end

disp(['Difference at i = ',num2str(num_iter),': ',num2str(norm(squeeze(Vi(:,:,end)-Vd(:,:,end))))])