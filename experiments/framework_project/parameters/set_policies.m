function [policies, names] = set_policies()
%SET_POLICIES select the policies
%   Comment/uncomment policies and corresponding names you want to discard/select
%
%   Output:
%       policies: cell array with function pointer to policies
%       names   : cell array with the policies names strings
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

policies = { ...
    @policy_ucb1 ...
    , @policy_ucb1_m ...
    , @policy_ucb_l ...
    , @policy_ucb_lm ...
    };

names = { ...
    'ucb1' ...
    , 'ucb1-m' ...
    , 'ucb-l' ...
    , 'ucb-lm' ...
    };
