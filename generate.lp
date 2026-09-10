% agent movement
action(0,T) | action(1,T) | action(2,T) | action(3,T) | action(4,T) :- agent(X,Y,T), T<horizon.
agent(X+1,Y,T+1) :- agent(X,Y,T), action(0,T).    % move right
agent(X,Y+1,T+1) :- agent(X,Y,T), action(1,T).    % move up
agent(X-1,Y,T+1) :- agent(X,Y,T), action(2,T).    % move left
agent(X,Y-1,T+1) :- agent(X,Y,T), action(3,T).    % move down
agent(X,Y,T+1) :- agent(X,Y,T), action(4,T).    % do nothing

% compute reward
reward(@compute_reward_new(LAWN,LAKE,D_LAWN,D_LAKE),T) :- sips_lake(LAKE,T), mows_lawn(LAWN,T), dist_grass(D_LAWN,T), dist_lake(D_LAKE,T), T>0.

% q-learning features
sips_lake(1,T) :- lake_timer(I,lake_respawn,T+1).
sips_lake(0,T) :- lake_timer(_,_,T), not sips_lake(1,T).
mows_lawn(1,T) :- grass_timer(I,grass_respawn,T+1).
mows_lawn(0,T) :- grass_timer(_,_,T), not mows_lawn(1,T).
dist_lake(D,T) :- agent(X,Y,T), pref_lake(X,Y,T,I), lake_dist(X,Y,D,I).
dist_lake(0,T) :- agent(X,Y,T), pref_lake(X,Y,T,-1).
dist_grass(D,T) :- agent(X,Y,T), pref_grass(X,Y,T,I), grass_dist(X,Y,D,I).
dist_grass(0,T) :- agent(X,Y,T), pref_grass(X,Y,T,-1).

% lakes
lake_agent_prox(I,T) :- lake(X1,Y1,I), agent(X2,Y2,T), |X1 - X2| + |Y1 - Y2| == 1.
lake_timer(I,C-1,T+1) :- lake_timer(I,C,T), C!=0, T<=horizon.
lake_timer(I,0,T+1) :- lake_timer(I,0,T), T<=horizon, not lake_agent_prox(I,T).
lake_timer(I,lake_respawn,T+1) :- lake_timer(I,0,T), lake_agent_prox(I,T), T<=horizon.

%grass
grass_timer(I,C-1,T+1) :- grass_timer(I,C,T), C!=0, T<=horizon.
grass_timer(I,0,T+1) :- grass_timer(I,0,T), grass(X,Y,I), T<=horizon, not agent(X,Y,T).
grass_timer(I,grass_respawn,T+1) :- grass_timer(I,0,T), grass(X,Y,I), agent(X,Y,T), T<=horizon.

pref(X,Y,T,A) :- lake_order(X,Y,I,O), lake_timer(I,0,T), lake_action(X,Y,I,A), not has_smaller_active(X,Y,I,T).
pref(X,Y,T,-1) :- lake_order(X,Y,I,_), lake_timer(I,_,T), not any_lake_active(T).
pref_lake(X,Y,T,I) :- lake_order(X,Y,I,O), lake_timer(I,0,T), not has_smaller_active(X,Y,I,T).
pref_lake(X,Y,T,-1) :- lake_order(X,Y,I,_), lake_timer(I,_,T), not any_lake_active(T).
any_lake_active(T) :- lake_timer(_,0,T).
has_smaller_active(X,Y,I,T) :- lake_timer(I, 0, T), lake_order(X, Y, I, O), lake_timer(I_Other, 0, T), lake_order(X, Y, I_Other, O_Other),O_Other < O.
has_smaller_grass(X,Y,I,T) :- grass_timer(I, 0, T), grass_order(X, Y, I, O), grass_timer(I_Other, 0, T), grass_order(X, Y, I_Other, O_Other),O_Other < O.
any_grass_active(T) :- grass_timer(_,0,T).
pref_grass(X,Y,T,I) :- grass_order(X,Y,I,O), grass_timer(I,0,T), not has_smaller_grass(X,Y,I,T).
pref_grass(X,Y,T,-1) :- grass_order(X,Y,I,_), grass_timer(I,_,T), not any_grass_active(T).

% can't leave the grid
:- agent(X,Y,T), X < 0.
:- agent(X,Y,T), X > size - 1.
:- agent(X,Y,T), Y < 0.
:- agent(X,Y,T), Y > size - 1.

% can't go through walls or lakes
:- agent(X,Y,T), wall(X,Y).
:- agent(X,Y,T), lake(X,Y,I).

% compute frog action w/ sampled lookup table
% todo: what about rounding errors?
valid_act_pos(X,Y,A,O,N) :- act_pos(X,Y,A,O,N), N > 1.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,AP), valid_act_pos(X,Y,A,O,N), not any_frog_timer(I,T,W), not frog_capture(I,T,W), A < AP, R >= O * (30 / (N-1)), R < (O + 1) * (30 / (N-1)).
f_action(AP,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,AP), valid_act_pos(X,Y,AP,O,N), not any_frog_timer(I,T,W), not frog_capture(I,T,W), R >= O * (30 / (N-1)), R < O * (30 / (N-1)) + 70.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), act_pos(X,Y,A,O,1), not any_frog_timer(I,T,W), not frog_capture(I,T,W).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,AP), valid_act_pos(X,Y,A,O,N), not any_frog_timer(I,T,W), not frog_capture(I,T,W), A > AP, AP != -1, R >= 70 + (O - 1) * (30 / (N-1)), R < 70 + O * (30 / (N-1)).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref(X,Y,T,-1), valid_act_pos(X,Y,A,O,N), not any_frog_timer(I,T,W), not frog_capture(I,T,W), R >= O * (100 / N), R < (O + 1) * (100 / N).
frog(X,Y,I,T+1,W) :- frog(X,Y,I,T,W), any_frog_timer(I,T,W), not frog_capture(I,T,W).

% compute frog movement
frog(X+1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(0,I,T,W).    % move right
frog(X,Y+1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(1,I,T,W).    % move up
frog(X-1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(2,I,T,W).    % move left
frog(X,Y-1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(3,I,T,W).    % move down

%%% OPTIMIZATION %%%

% try not to step onto frogs
:~ norm_violation(W). [1@ horizon+1,W]

#maximize {R@ horizon+1-T,T : reward(R,T)}.
#show action/2.
#defined f_rnd/4.
#defined wall/2.
#defined lake/2.
#defined act_pos/5.
#defined lake_order/4.
#defined lake_action/4.
#defined frog_timer/3.
#defined ctd/3.
#defined ctd_world/1.
#defined sips_lake/2.
