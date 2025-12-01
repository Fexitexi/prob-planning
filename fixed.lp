% agent movement
action(0,T) | action(1,T) | action(1,T) | action(3,T) | action(4,T) :- agent(X,Y,T), T<horizon.
agent(X+1,Y,T) :- agent(X,Y,T), action(0,T).    % move right
agent(X,Y+1,T) :- agent(X,Y,T), action(1,T).    % move up
agent(X-1,Y,T) :- agent(X,Y,T), action(2,T).    % move left
agent(X,Y-1,T) :- agent(X,Y,T), action(3,T).    % move down
agent(X,Y,T) :- agent(X,Y,T), action(4,T).    % do nothing

% can't leave the grid
:- agent(X,Y,T), X < 0.
:- agent(X,Y,T), X > size - 1.
:- agent(X,Y,T), Y < 0.
:- agent(X,Y,T), Y > size - 1.

% can't go through walls or lakes
:- agent(X,Y,T), wall(X,Y).
:- agent(X,Y,T), lake(X,Y,I).

% timer
grass_timer(I,C-1,T+1) :- grass_timer(I,C,T), C!=0, T<horizon.
grass_timer(I,0,T+1) :- grass_timer(I,0,T), grass(X,Y,I), not agent(X,Y,T).
grass_timer(I,grass_respawn,T+1) :- grass_timer(I,0,T), grass(X,Y,I), agent(X,Y,T).
lake_agent_prox(I,T) :- lake(X1,Y1,I), agent(X2,Y2,T), |X1 - X2| <= 1, |Y1 - Y2| <= 1.
lake_timer(I,C-1,T+1) :- lake_timer(I,C,T), C!=0, T<horizon.
lake_timer(I,0,T+1) :- lake_timer(I,0,T), not lake_agent_prox(I,T).
lake_timer(I,lake_respawn,T+1) :- lake_timer(I,0,T), lake_agent_prox(I,T).


%%% OPTIMIZATION %%%

% try not to step onto frogs
:~ agent(X,Y,T), frog(X,Y,I,T,W). [1@1]