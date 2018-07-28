function [div, x] = max_kldiv(p,lim,tol)
%MAX_KLDIV maximizes the kl divergense under a constraint that KL(p,x) \leq lim
%   Input:
%       p  : real value for the expected probability of the bernoulli
%           variable
%       lim: limit for the KL divergence
%       tol: tolerance for the limit excess
%
%   Output:
%       div: value of the KL divergence
%       x  : value of x s.t. the divergence is maximized and the limit is
%           satisfied with the set tolerance
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

p = max(p, tol);
p = min(p, 1-tol);
a = p;
b = 1-tol;

div = 1;

div_a = p * log(p / a) + (1 - p) * log((1-p) / (1-a)) - lim;
assert(div_a < 0);

div_b = p * log(p / b) + (1 - p) * log((1-p) / (1-b)) - lim;

if div_a * div_b < 0
    max_iter = 1;
    
    while abs(div) > tol && max_iter < 10000
        x = (a + b) / 2;
        div = p * log(p / x) + (1 - p) * log((1-p) / (1-x)) - lim;
        
        if div * div_a > 0
            a = x;
        else
            b = x;
        end
        
        max_iter = max_iter + 1;
    end
    
else
    x = b;
end