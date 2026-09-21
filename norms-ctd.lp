norm_violation(W) :- frog_capture(F,T,W), not ctd(F,T,W).
norm_violation(W) :- ctd(F,T,W), not ctd_check(F,W).
norm_violation(W) :- ctd(F,T,W).
