#program state.
valid_act_pos(X,Y,A,O,N) :- act_pos(X,Y,A,O,N), N > 1.

#program agent_movement.

%agent movement
agent(X+1,Y,T+1) :- agent(X,Y,T), action(0,T).    % move right
agent(X,Y+1,T+1) :- agent(X,Y,T), action(1,T).    % move up
agent(X-1,Y,T+1) :- agent(X,Y,T), action(2,T).    % move left
agent(X,Y-1,T+1) :- agent(X,Y,T), action(3,T).    % move down
agent(X,Y,T+1) :- agent(X,Y,T), action(4,T).    % do nothing

% lakes
lake_agent_prox(I,T) :- lake(X1,Y1,I), agent(X2,Y2,T), |X1 - X2| + |Y1 - Y2| == 1.
lake_timer(I,C-1,T+1) :- lake_timer(I,C,T), C!=0, T<=horizon.
lake_timer(I,0,T+1) :- lake_timer(I,0,T), T<=horizon, not lake_agent_prox(I,T).
lake_timer(I,lake_respawn,T+1) :- lake_timer(I,0,T), lake_agent_prox(I,T), T<=horizon.

% frog movement navigation
pref(X,Y,T,A) :- lake_order(X,Y,I,O), lake_timer(I,0,T), lake_action(X,Y,I,A), not has_smaller_active(X,Y,I,T).
pref(X,Y,T,-1) :- lake_order(X,Y,I,_), lake_timer(I,_,T), not any_lake_active(T).
any_lake_active(T) :- lake_timer(_,0,T).
has_smaller_active(X,Y,I,T) :- lake_timer(I, 0, T), lake_order(X, Y, I, O), lake_timer(I_Other, 0, T), lake_order(X, Y, I_Other, O_Other),O_Other < O.

#program frog_movement.

% action selection
frog(X,Y,I,T+1) :- frog(X,Y,I,T), any_frog_timer(I,T), not frog_capture(I,T).

% compute frog movement
frog(X+1,Y,I,T+1) :- frog(X,Y,I,T), f_action(0,I,T).    % move right
frog(X,Y+1,I,T+1) :- frog(X,Y,I,T), f_action(1,I,T).    % move up
frog(X-1,Y,I,T+1) :- frog(X,Y,I,T), f_action(2,I,T).    % move left
frog(X,Y-1,I,T+1) :- frog(X,Y,I,T), f_action(3,I,T).    % move down

frog_capture(I,T) :- agent(X,Y,T), frog(X,Y,I,T).
ctd_check(F) :- frog_capture(F,T), ctd(F,T).

lake_frog_prox(F,L,T) :- lake(X1,Y1,L), frog(X2,Y2,F,T), |X1 - X2| + |Y1 - Y2| == 1.
lake_frog_prox(F,L,T) :- lake(X1,Y1,L), frog(X2,Y2,F,T), |X1 - X2| + |Y1 - Y2| == 2, |X1 - X2| == 1.
lake_frog_prox(F,L,T) :- lake(X1,Y1,L), frog(X2,Y2,F,T), |X1 - X2| + |Y1 - Y2| == 2, |Y1 - Y2| == 1.
ctd_init(F,T) :- lake_timer(L,0,T), lake_frog_prox(F,L,T), lake_agent_prox(L,T), agent(X1,Y1,T), frog(X2,Y2,F,T), X1 != X2.
ctd_init(F,T) :- lake_timer(L,0,T), lake_frog_prox(F,L,T), lake_agent_prox(L,T), agent(X1,Y1,T), frog(X2,Y2,F,T), Y1 != Y2.
ctd(F,T) :- ctd_init(F,TI), T = TI..TI + 4, TI <= horizon.
any_frog_timer(F,T) :- ctd_init(F,TI), T = TI..TI + 5, TI <= horizon.

ctd(I,T) :- ctd(I,T), frog(X,Y,I,T).
any_frog_timer(I,T) :- frog_timer(I,T), frog(X,Y,I,T).

norm_violation() :- frog_capture(F,T), not ctd(F,T).
norm_violation() :- ctd(F,T), not ctd_check(F).
norm_violation() :- ctd(F,T).

#show norm_violation/0.
#defined norm_violation/0.
#defined frog_timer/2.
#defined act_pos/5.
#defined lake_order/4.
#defined lake_action/4.
#defined frog_timer/3.
#defined ctd/2.
#defined ctd_world/1.
#defined lake_timer/3.
#defined action/2.
