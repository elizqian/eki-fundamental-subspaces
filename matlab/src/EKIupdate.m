function [Vnext, problem] = EKIupdate(Vnow,problem,obs,method)

[d,J] = size(Vnow);

switch obs
    case 'deterministic'
        m = problem.m;
    case 'stochastic'
        noise = mvnrnd(zeros(1,problem.n),problem.Sigma,J)';
        m = problem.m + noise;
end


switch(method)
    case 'uses-adjoints'
        mu_i = mean(Vnow,2);
        Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
        S_i = (problem.H*Gam_i*problem.H' + problem.Sigma);
        K_i = Gam_i*problem.H'/S_i;
        Vnext = (eye(d) - K_i*problem.H)*Vnow + K_i*m;

    case 'adjoint-free'
        Znow = [Vnow; problem.H*Vnow];
        H = [zeros(problem.n,problem.d), eye(problem.n)];
        mu_i = mean(Znow,2);
        Gam_i = (Znow-mu_i)*(Znow-mu_i)'/(J-1);
        S_i = (H*Gam_i*H' + problem.Sigma);
        K_i = Gam_i*H'/S_i;
        Znext = (eye(d+problem.n) - K_i*H)*Znow + K_i*m;
        Vnext = Znext(1:d,:);
end