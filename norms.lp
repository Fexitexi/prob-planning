norm_violation(W) :- frog_capture(F,T,W), not any_ctd(F,T,W).
norm_violation(W) :- any_ctd(F,T,W), not ctd_check(F,W).


frog_capture(I,T,W) :- agent(X,Y,T), frog(X,Y,I,T,W).

any_ctd(F,T,W) :- ctd(F,L,T,W).
ctd_check(F,W) :- frog_capture(F,T,W), any_ctd(F,T,W).


lake_frog_prox(F,L,T,W) :- lake(X1,Y1,L), frog(X2,Y2,F,T,W), |X1 - X2| + |Y1 - Y2| == 1.
lake_frog_prox(F,L,T,W) :- lake(X1,Y1,L), frog(X2,Y2,F,T,W), |X1 - X2| + |Y1 - Y2| == 2, |X1 - X2| == 1.
lake_frog_prox(F,L,T,W) :- lake(X1,Y1,L), frog(X2,Y2,F,T,W), |X1 - X2| + |Y1 - Y2| == 2, |Y1 - Y2| == 1.
ctd_init(F,L,T,W) :- lake_timer(L,0,T), lake_frog_prox(F,L,T,W), lake_agent_prox(L,T), agent(X1,Y1,T), frog(X2,Y2,F,T,W), X1 != X2.
ctd_init(F,L,T,W) :- lake_timer(L,0,T), lake_frog_prox(F,L,T,W), lake_agent_prox(L,T), agent(X1,Y1,T), frog(X2,Y2,F,T,W), Y1 != Y2.
ctd(F,L,T,W) :- ctd_init(F,L,TI,W), T = TI..TI + 4, TI <= horizon.
any_frog_timer(F,T,W) :- ctd(F,L,T,W).
any_frog_timer(I,T,W) :- frog_timer(I,D,T), frog(X,Y,I,T,W).


% from outside
ctd_init(F,L,TI-4,W) :- frog(X,Y,F,0,W), ctd(F,L,TI).

#show ctd_init/4.
#defined ctd/2.
