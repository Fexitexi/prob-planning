% agent movement
agent(X+1,Y,T+1) :- agent(X,Y,T), action(0,T).    % move right
agent(X,Y+1,T+1) :- agent(X,Y,T), action(1,T).    % move up
agent(X-1,Y,T+1) :- agent(X,Y,T), action(2,T).    % move left
agent(X,Y-1,T+1) :- agent(X,Y,T), action(3,T).    % move down
agent(X,Y,T+1) :- agent(X,Y,T), action(4,T).    % do nothing

% lakes
lake_frog_prox(I,T,W) :- lake(X1,Y1,I), frog(X2,Y2,I,T,W), |X1 - X2| + |Y1 - Y2| == 1.
lake_agent_prox(I,T) :- lake(X1,Y1,I), agent(X2,Y2,T), |X1 - X2| + |Y1 - Y2| == 1.
lake_timer(I,C-1,T+1) :- lake_timer(I,C,T), C!=0, T<horizon.
lake_timer(I,0,T+1) :- lake_timer(I,0,T), T<horizon, not lake_agent_prox(I,T).
lake_timer(I,lake_respawn,T+1) :- lake_timer(I,0,T), lake_agent_prox(I,T), T<horizon.

pref(X,Y,T,A) :- lake_order(X,Y,I,O), lake_timer(I,0,T), lake_action(X,Y,I,A), not has_smaller_active(X,Y,I,T).
pref(X,Y,T,-1) :- lake_order(X,Y,I,_), lake_timer(I,_,T), not any_lake_active(T).
any_lake_active(T) :- lake_timer(_,0,T).
has_smaller_active(X,Y,I,T) :- lake_timer(I, 0, T), lake_order(X, Y, I, O), lake_timer(I_Other, 0, T), lake_order(X, Y, I, O_Other),O_Other < O.

% compute frog action w/ sampled lookup table
frog_timer(I,D,0,W) :- frog(X,Y,I,0,W), frog_timer(I,D,0).
frog_timer(I,D-1,T+1,W) :- frog_timer(I,D,T,W), D > 0.
any_frog_timer(I,T,W) :- frog_timer(I,D,T,W).
valid_act_pos(X,Y,A,O,N) :- act_pos(X,Y,A,O,N), N > 1.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,AP), valid_act_pos(X,Y,A,O,N), not any_frog_timer(I,T,W), A < AP, R >= O * (30 / (N-1)), R < (O + 1) * (30 / (N-1)).
f_action(AP,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,AP), valid_act_pos(X,Y,AP,O,N), not any_frog_timer(I,T,W), R >= O * (30 / (N-1)), R < O * (30 / (N-1)) + 70.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), act_pos(X,Y,A,O,1), not any_frog_timer(I,T,W).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,AP), valid_act_pos(X,Y,A,O,N), not any_frog_timer(I,T,W), A > AP, R >= 70 + (O - 1) * (30 / (N-1)), R < 70 + O * (30 / (N-1)).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,-1), valid_act_pos(X,Y,A,O,N), not any_frog_timer(I,T,W), R >= O * (100 / N), R < (O + 1) * (100 / N).
frog(X,Y,I,T+1,W) :- frog(X,Y,I,T,W), any_frog_timer(I,T,W).

% compute frog movement
frog(X+1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(0,I,T,W).    % move right
frog(X,Y+1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(1,I,T,W).    % move up
frog(X-1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(2,I,T,W).    % move left
frog(X,Y-1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(3,I,T,W).    % move down


%%% OPTIMIZATION %%%

norm_violation(W) :- agent(X,Y,T), frog(X,Y,I,T,W), not ctd(I,_,0,W).
norm_violation(W) :- ctd(I,D,0,W), not ctd_check(W).

ctd_check(W) :- agent(X,Y,T), frog(X,Y,I,T,W), ctd(I,D,T,W).

ctd(I,D-1,T+1,W) :- ctd(I,D,T,W), D > 0.
ctd(I,D,0,W) :- frog(X,Y,I,T,W), ctd(I,D,0).

#show norm_violation/1.
%#show ctd/1.
#defined f_rnd/4.
#defined act_pos/5.
#defined lake_order/4.
#defined lake_action/4.
#defined frog_timer/3.
#defined ctd/3.
#defined ctd/2.
#defined ctd/1.
#defined ctd_world/1.