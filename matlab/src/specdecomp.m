function spdc = specdecomp(H,vvi,Sigma)
[n,d] = size(H);

fisher = H'*(Sigma\H);
Hplus  = pinv(fisher)*(H'/Sigma);

Gam = cov(vvi');
HGamH = H*Gam*H';
%% measurement space eigenvalue problem and projectors
% solve generalized eigenvalue problem and sort eigenvalues and vectors
[W,delta] = eig(HGamH,Sigma);
[delta,ind] = sort(diag(delta),'descend');
r = sum(abs(delta)>1e-10);
h = rank(H);
W = W(:,ind);

% normalize first r eigenvectors (the latter n-r will be overwritten)
for i = 1:r
    W(:,i) = W(:,i)/sqrt(W(:,i)'*Sigma*W(:,i));
end

% compute bases for complementary subspaces defined in Section 3.1
basisSigInvH = orth(Sigma\H);
basisKerHT = null(H');

% initialize random vectors to be orthogonalized
temp = rand(n,n-r);
for ell = r+1:n
    if ell <= h     % if within first h, restrict to component in Ran(SigInvH)
        w = basisSigInvH*basisSigInvH'*temp(:,ell-r); 
    else            % if in last n-h, restrict to component in Ker(H')
        w = basisKerHT*basisKerHT'*temp(:,ell-r);     
    end
    for k = 1:ell-1 % GS orthogonalize                                   
        w = w - (w'*Sigma*W(:,k))/(W(:,k)'*Sigma*W(:,k))*W(:,k);
    end
    W(:,ell) = w/sqrt(w'*Sigma*w);  % normalize w.r.t. Sigma inner product
end

spdc.W = W;
spdc.delta = delta;
spdc.calP = Sigma*W(:,1:r)*W(:,1:r)';
if h>r
    spdc.calQ = Sigma*W(:,r+1:h)*W(:,r+1:h)';
else
    spdc.calQ = zeros(n,n);
end
spdc.calN = eye(n) - spdc.calP - spdc.calQ;

%% state space eigenvector definition and projectors

% define leading h eigenvectors in terms of leading h measurement eigvecs
U = zeros(d,d); 
for ell = 1:h 
    if ell <= r
        U(:,ell) = Gam*H'*W(:,ell)/delta(ell);
    else
        U(:,ell) = Hplus*Sigma*W(:,ell);
    end
end

spdc.U = U;
spdc.bbP = U(:,1:r)*U(:,1:r)'*fisher;
if h > r
    spdc.bbQ = U(:,r+1:h)*U(:,r+1:h)'*fisher;
else
    spdc.bbQ = zeros(d,d);
end
spdc.bbN = eye(d) - spdc.bbP - spdc.bbQ;

