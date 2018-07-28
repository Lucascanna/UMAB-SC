%% README

Policies should have the following signature:
    [idx_best_arm, opt_policies] = POLICY_NAME(arms, p0, p1, t, opt_policies)
where the only choice is the POLICY_NAME, you have to add to SET_POLICIES