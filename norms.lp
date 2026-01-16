norm_violation(W) :- frog_capture(F,T,W), not any_ctd(F,T,W).
norm_violation(W) :- any_ctd(F,T,W), not ctd_check(F,W).


frog_capture(I,T,W) :- agent(X,Y,T), frog(X,Y,I,T,W).

any_ctd(F,T,W) :- ctd(F,L,T,W).
ctd_check(F,W) :- frog_capture(F,T,W), any_ctd(F,T,W).

ctd_init(F,L,T,W) :- lake_timer(L,0,T), lake_frog_prox(F,L,T,W), lake_agent_prox(L,T).
ctd(F,L,T,W) :- ctd_init(F,L,TI,W), T = TI..TI + 4, TI <= horizon.
any_frog_timer(F,T,W) :- ctd(F,L,T,W).

% from outside
ctd_init(F,L,TI-4,W) :- frog(X,Y,F,0,W), ctd(F,L,TI).

#show ctd_init/4.
#defined ctd/2.
