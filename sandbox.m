load gevs

% my expressions
test = zeros(size(gevs));
test(:,1) = gevs(:,1);
fac = 0;
for i = 1:size(gevs,2)-1
[test(:,i+1), fac] = recur(test(:,i),fac);
end


% pavlos expressions
pred1 = @(x1) x1./(1+x1);
pred2 = @(x1) x1.*((1+3*x1)./((1+x1).^2));
pred3 = @(x1, x2) (x2.*(1+x2+x1.*(5+x2)))./((1+x1).*((1+x2).^2));

ps_delta2 = zeros(size(gevs));
ps_delta2(:,1) = gevs(:,1);
ps_delta2(:,2) = pred1(ps_delta2(:,1));
ps_delta2(:,3) = pred2(ps_delta2(:,2));
ps_delta2(:,4) = pred3(ps_delta2(:,2),ps_delta2(:,3));

function [lambda,factor] = recur(lam,fac)
    lambda = lam./(1+lam).*(1+2*fac);
    factor = 1./(1+lambda).*lam./(1+lam) + fac./(1+lambda);
end