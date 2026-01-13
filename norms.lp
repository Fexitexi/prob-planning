norm_violation(W) :- frog_capture(I,T,W), not ctd(I,T,W).
norm_violation(W) :- ctd(I,T,W), not ctd_check(W).
norm_violation(W) :- lake_frog_prox(I,T,W), lake_agent_prox(I,T), not ctd_check(W).

frog_capture(I,T,W) :- agent(X,Y,T), frog(X,Y,I,T,W).

ctd_check(W) :- frog_capture(I,T,W), ctd(I,T,W).

ctd_init(I,T,W) :- lake_timer(I,lake_respawn,T), lake_frog_prox(I,T,W), lake_agent_prox(I,T).
ctd(I,T,W) :- ctd_init(I,TI,W), T = TI..TI + 4.
any_frog_timer(I,T,W) :- ctd(I,T,W).
any_frog_timer(I,T+1,W) :- any_frog_timer(I,T,W), T < horizon.

% from outside
ctd_init(I,TI-4,W) :- frog(X,Y,I,0,W), ctd(I,TI).

#show ctd/3.
#defined ctd/2.
