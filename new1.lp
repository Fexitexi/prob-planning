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
:- agent(X,Y,T), lake(X,Y,I).

% compute frog action w/ sampled lookup table
f_action(A,I,T,W) :- f_rnd(I,T,W,R), frog(X,Y,I,T,W), act_pos(X,Y,A,O,N), R >= O * (100 / N), R < (O + 1) * (100 / N).

% compute frog movement
frog(X+1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(0,I,T,W).    % move right
frog(X,Y+1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(1,I,T,W).    % move up
frog(X-1,Y,I,T+1,W) :- frog(X,Y,I,T,W), f_action(2,I,T,W).    % move left
frog(X,Y-1,I,T+1,W) :- frog(X,Y,I,T,W), f_action(3,I,T,W).    % move down


%%% OPTIMIZATION %%%

% try not to step onto frogs
:~ agent(X,Y,T), frog(X,Y,I,T,W). [1@ horizon +1]

#maximize {R@ horizon-T : action_reward(R,T)}.
#show history/2.
#show action_reward/2.
#show action/2.