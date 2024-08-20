function [H, Sigma, v0] = EKIsetupRandom(n,d,J)
% sets up a random least squares problem and initial ensemble so that each
% fundamental subspace is non-trivial

% generate random H and subtract one direction from both rowspace and
% columnspace so it and its transpose both have non-trivial kernel
H = rand(n,d);
v = rand(d,1);
v = v/norm(v);      
H = H - H * (v*v'); % v is a unit vector in Ker(H) now
w = rand(n,1);
w = w/norm(w);
H = H - (w*w')*H;   % w is a unit vector in Ker(H') now

% generate random positive definite Sigma
L = 0.5*rand(n);
Sigma = L*L';

% generate random ensemble with nonzero component in both Ker(H) and
% Ran(H'), but does not span all of Ran(H')
[basisH,~,~] = svd(H');
basisV = [v, basisH(:,2:d-2)]; % basis for particles consisting of something in Ker(H) and not the entirety of Ran(H')
v0 = basisV * rand(d-2,J);
