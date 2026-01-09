% agent movement
agent(X+1,Y,T+1) :- agent(X,Y,T), action(0,T).    % move right
agent(X,Y+1,T+1) :- agent(X,Y,T), action(1,T).    % move up
agent(X-1,Y,T+1) :- agent(X,Y,T), action(2,T).    % move left
agent(X,Y-1,T+1) :- agent(X,Y,T), action(3,T).    % move down
agent(X,Y,T+1) :- agent(X,Y,T), action(4,T).    % do nothing

% compute frog action w/ sampled lookup table
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,T,AP), act_pos(X,Y,A,O,N'), N = N' - 1, N' > 1, A < AP, R >= O * (30 / N), R < (O + 1) * (30 / N).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,T,AP), act_pos(X,Y,A,O,N'), N = N' - 1, N' > 1, A = AP, R >= O * (30 / N), R < O * (30 / N) + 70.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,T,AP), act_pos(X,Y,A,O,N'), N = N' - 1, N' = 1, A = AP.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,T,AP), act_pos(X,Y,A,O,N'), N = N' - 1, N' > 1, A > AP, R >= 70 + (O - 1) * (30 / N), R < 70 + O * (30 / N).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,T,-1), act_pos(X,Y,A,O,N), R >= O * (100 / N), R < (O + 1) * (100 / N).

% compute frog movement
frog(X+1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(0,I,T,W).    % move right
frog(X,Y+1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(1,I,T,W).    % move up
frog(X-1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(2,I,T,W).    % move left
frog(X,Y-1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(3,I,T,W).    % move down


%%% OPTIMIZATION %%%

norm_violation(W) :- agent(X,Y,T), frog(X,Y,I,T,W).

#show norm_violation/1.