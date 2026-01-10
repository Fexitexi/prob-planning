% agent movement
action(0,T) | action(1,T) | action(2,T) | action(3,T) | action(4,T) :- agent(X,Y,T), T<horizon.
agent(X+1,Y,T+1) :- agent(X,Y,T), action(0,T).    % move right
agent(X,Y+1,T+1) :- agent(X,Y,T), action(1,T).    % move up
agent(X-1,Y,T+1) :- agent(X,Y,T), action(2,T).    % move left
agent(X,Y-1,T+1) :- agent(X,Y,T), action(3,T).    % move down
agent(X,Y,T+1) :- agent(X,Y,T), action(4,T).    % do nothing

% movement history
history(H,0) :- action(A,0), H = 10 + A.
history(H,T) :- history(H',T-1), action(A,T), H = H' * 10 + A.

% compute reward
action_reward(@compute_reward(H), T) :- history(H, T).

% can't leave the grid
:- agent(X,Y,T), X < 0.
:- agent(X,Y,T), X > size - 1.
:- agent(X,Y,T), Y < 0.
:- agent(X,Y,T), Y > size - 1.

% can't go through walls or lakes
:- agent(X,Y,T), wall(X,Y).
:- agent(X,Y,T), lake(X,Y).

% compute frog action w/ sampled lookup table
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,H,AP), history(H,T), act_pos(X,Y,A,O,N'), N = N' - 1, N' > 1, A < AP, R >= O * (30 / N), R < (O + 1) * (30 / N).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,H,AP), history(H,T), act_pos(X,Y,A,O,N'), N = N' - 1, N' > 1, A = AP, R >= O * (30 / N), R < O * (30 / N) + 70.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,H,AP), history(H,T), act_pos(X,Y,A,O,N'), N = N' - 1, N' = 1, A = AP.
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,H,AP), history(H,T), act_pos(X,Y,A,O,N'), N = N' - 1, N' > 1, A > AP, R >= 70 + (O - 1) * (30 / N), R < 70 + O * (30 / N).
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), pref_act(X,Y,H,-1), history(H,T), act_pos(X,Y,A,O,N), R >= O * (100 / N), R < (O + 1) * (100 / N).

% compute frog movement
frog(X+1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(0,I,T,W).    % move right
frog(X,Y+1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(1,I,T,W).    % move up
frog(X-1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(2,I,T,W).    % move left
frog(X,Y-1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(3,I,T,W).    % move down

%%% OPTIMIZATION %%%

% try not to step onto frogs
norm_violation(W) :- agent(X,Y,T), frog(X,Y,I,T,W).
:~ norm_violation(W). [1@ horizon +1]

#maximize {R@ horizon-T : action_reward(R,T)}.
#show frog/5.
#show action/2.
#defined f_rnd/4.
#defined wall/2.
#defined lake/2.
