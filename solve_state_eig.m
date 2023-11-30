function [sig2,U,M,Vhat,Lam,What] = solve_state_eig(V,problem)

Gam = cov(V');
[U,Sig2] = eig(Gam*problem.fisher);
sig2 = diag(Sig2);
[sig2,ind] = sort(sig2,'descend');
U = U(:,ind);

[Vhat,M] = eig(Gam);
[M,ind] = sort(diag(M),'descend');
M = diag(M);
Vhat = Vhat(:,ind);

[What,Lam] = eig(problem.fisher);
[Lam,ind] = sort(diag(Lam),'descend');
What = What(:,ind);