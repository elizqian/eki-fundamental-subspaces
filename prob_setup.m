function problem = prob_setup(n,testcase)

switch(testcase)
    case 'id-sq'
        G = eye(n);
        Geps = 0.01*eye(n);
        mupr = ones(n,1);
        Gpr = ((2*eye(n)-diag(ones(n-1,1),-1)-diag(ones(n-1,1),1))*(n-1)+1*eye(n))^-2;
        truth = mvnrnd(mupr,Gpr,1)';
        meas  = G*truth + mvnrnd(zeros(n,1),Geps)';
        x = linspace(0,1,n)';

    case 'id-sparse'
        ind = round([0.25 0.5 0.75]*n);
        G = eye(n);
        G = G(ind,:);
        Geps = 0.01*eye(3);
        mupr = ones(n,1);
        Gpr = ((2*eye(n)-diag(ones(n-1,1),-1)-diag(ones(n-1,1),1))*(n-1)+1*eye(n))^-2;
        truth = mvnrnd(mupr,Gpr,1)';
        meas  = G*truth + mvnrnd(zeros(3,1),Geps)';
        x = linspace(0,1,n)';
    case 'id-test'
        G = eye(n);
        G = G(1:3,:);
        Geps = 0.01*eye(3);
        mupr = ones(n,1);
        Gpr = ((2*eye(n)-diag(ones(n-1,1),-1)-diag(ones(n-1,1),1))*(n-1)+1*eye(n))^-2;
        truth = mvnrnd(mupr,Gpr,1)';
        meas  = G*truth + mvnrnd(zeros(3,1),Geps)';
        x = linspace(0,1,n)';
    case 'id-overdet'
        G = [eye(n); eye(n)];
        Geps = 0.01*eye(2*n);
        mupr = ones(n,1);
        Gpr = ((2*eye(n)-diag(ones(n-1,1),-1)-diag(ones(n-1,1),1))*(n-1)+1*eye(n))^-2;
        truth = mvnrnd(mupr,Gpr,1)';
        meas  = G*truth + mvnrnd(zeros(2*n,1),Geps)';
        x = linspace(0,1,n)';
    
    case 'rnd-sq'
        G = rand(n);
        Geps = 0.01*eye(n);
        mupr = ones(n,1);
        Gpr = ((2*eye(n)-diag(ones(n-1,1),-1)-diag(ones(n-1,1),1))*(n-1)+1*eye(n))^-2;
        truth = mvnrnd(mupr,Gpr,1)';
        meas  = G*truth + mvnrnd(zeros(n,1),Geps)';
        x = linspace(0,1,n)';

    case 'rnd-sparse'
        G = rand(3,n);
        Geps = 0.01*eye(3);
        mupr = ones(n,1);
        Gpr = ((2*eye(n)-diag(ones(n-1,1),-1)-diag(ones(n-1,1),1))*(n-1)+1*eye(n))^-2;
        truth = mvnrnd(mupr,Gpr,1)';
        meas  = G*truth + mvnrnd(zeros(3,1),Geps)';
        x = linspace(0,1,n)';
    case 'rnd-overdet'
        G = rand(2*n,n);
        Geps = 0.01*eye(2*n);
        mupr = ones(n,1);
        Gpr = ((2*eye(n)-diag(ones(n-1,1),-1)-diag(ones(n-1,1),1))*(n-1)+1*eye(n))^-2;
        truth = mvnrnd(mupr,Gpr,1)';
        meas  = G*truth + mvnrnd(zeros(2*n,1),Geps)';
        x = linspace(0,1,n)';
end
fisher = G'*(Geps\G);
Gposinv = G'*(Geps\G) + inv(Gpr);
mupos = Gposinv\(G'*(Geps\meas) + Gpr\mupr);
muLS  = pinv(fisher)*(G'*(Geps\meas));


problem.d = n;
problem.n = size(G,1);
problem.x = x;
problem.G = G;
problem.Geps = Geps;
problem.truth = truth;
problem.meas = meas;
problem.fisher = fisher;
problem.mupr = mupr;
problem.Gpr = Gpr;
problem.sample = @(m) mvnrnd(problem.mupr,problem.Gpr,m)';
problem.Gpos = inv(Gposinv);
problem.mupos = mupos;
problem.muLS  = muLS;
problem.Pi = pinv(fisher)*fisher;
