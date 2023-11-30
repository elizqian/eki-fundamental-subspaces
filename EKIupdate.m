function [Vnext, problem] = EKIupdate(Vnow,problem,obs,method)

[d,J] = size(Vnow);

if nargin < 3
    obs = 'a';
end

if nargin < 4
    method = 'dzh';
end

switch obs
    case 'a'
        m = problem.meas;
    case 'b'
        if ~isfield(problem,'noise9b')
            problem.noise9b = mvnrnd(zeros(1,problem.n),problem.Geps,J)';
        end
        m = problem.meas + problem.noise9b;
    case 'c'
        m = problem.meas + mvnrnd(zeros(1,problem.n),problem.Geps,J)';
end


switch(method)
    case 'dzh'
        mu_i = mean(Vnow,2);
        Gam_i = (Vnow-mu_i)*(Vnow-mu_i)'/(J-1);
        S_i = (problem.G*Gam_i*problem.G' + problem.Geps);
        K_i = Gam_i*problem.G'/S_i;
        Vnext = (eye(d) - K_i*problem.G)*Vnow + K_i*m;
        
    case 'iglesias'
        Znow = [Vnow; problem.G*Vnow];
        H = [zeros(problem.n,problem.d), eye(problem.n)];
        mu_i = mean(Znow,2);
        Gam_i = (Znow-mu_i)*(Znow-mu_i)'/(J-1);
        S_i = (H*Gam_i*H' + problem.Geps);
        K_i = Gam_i*H'/S_i;
        Znext = (eye(d+problem.n) - K_i*H)*Znow + K_i*m;
        Vnext = Znext(1:d,:);

end