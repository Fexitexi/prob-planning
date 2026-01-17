%norm_violation(W) :- frog_capture(F,T,W), not any_ctd(F,T,W).
%norm_violation(W) :- any_ctd(F,T,W), not ctd_check(F,W).

norm_violation(W) :- frog_capture(F,T,W).