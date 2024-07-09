function [Vnext, problem] = EKIupdate(Vnow,problem,obs,method)

[d,J] = size(Vnow);

if nargin < 3
    obs = 'a';
end

if nargin < 4
    method = 'dzh';
end

switch obs
    case 'det'
        m = problem.m;
    case 'b'
        if ~isfield(problem,'noise9b')
            problem.noise9b = mvnrnd(zeros(1,problem.n),problem.Sigma,J)';
        end
        m = problem.m + problem.noise9b;
    case 'stoch'
        m = problem.m + mvnrnd(zeros(1,problem.n),problem.Sigma,J)';
end


switch(method)
    case 'richardson'
        mu_i = mean(Vnow,2);
        Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
        S_i = (problem.H*Gam_i*problem.H' + problem.Sigma);
        K_i = Gam_i*problem.H'/S_i;

        if problem.iter <= 3
            Vnext = (eye(d) - K_i*problem.H)*Vnow + K_i*m;
        else
            Vnext = Vnow + 2*problem.iter*K_i*(m-problem.H*Vnow);
        end

    case 'dzh'
        mu_i = mean(Vnow,2);
        Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
        S_i = (problem.H*Gam_i*problem.H' + problem.Sigma);
        K_i = Gam_i*problem.H'/S_i;
        Vnext = (eye(d) - K_i*problem.H)*Vnow + K_i*m;

    case 'iglesias'
        Znow = [Vnow; problem.H*Vnow];
        H = [zeros(problem.n,problem.d), eye(problem.n)];
        mu_i = mean(Znow,2);
        Gam_i = (Znow-mu_i)*(Znow-mu_i)'/(J-1);
        S_i = (H*Gam_i*H' + problem.Sigma);
        K_i = Gam_i*H'/S_i;
        Znext = (eye(d+problem.n) - K_i*H)*Znow + K_i*m;
        Vnext = Znext(1:d,:);

    case 'stoch-simple'
        MtildeInv = (eye(d) + problem.Gi*problem.fisher);
        Ktilde = problem.Gi*problem.H'/(problem.H*problem.Gi*problem.H'+problem.Sigma);

        Vnext = MtildeInv\Vnow + Ktilde*m;
        problem.Gi = MtildeInv\problem.Gi;

end
problem.iter = problem.iter+1;