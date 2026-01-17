% external constants: frogs, horizon, radius, size

% player always starts at the 0 position
pcol(0,0).
prow(0,0).

% frogs can be anywhere within the window
#external fcol(F,C,0) : F=0..frogs-1, C=-radius..radius.
#external frow(F,R,0) : F=0..frogs-1, R=-radius..radius.

% if a frog is outside the current window, it will not be considered for
% the computation
#external foutside(F) : F=0..frogs-1.

% depending on where the window is, the cells can contain walls
#external wall(C,R) : C=-radius..radius, R=-radius..radius.

% rewards for the actions of the agent
#external action(A, S) : A=0..4, S=0..4, C=-radius..radius, R=-radius..radius.


% movement of the agent
pmove(0, T) | pmove(1, T) | pmove(2, T) :- T=1..horizon - 1.
prow(R + 1,T) | prow(R - 1,T) :- prow(R,T-1), pmove(0, T).
pcol(C,T) :- pcol(C,T-1), pmove(0, T).
pcol(C,T) :- pcol(C,T-1), pmove(2, T).
pcol(C + 1,T) | pcol(C - 1,T) :- pcol(C,T-1), pmove(1, T).
prow(R,T) :- prow(R,T-1), pmove(1, T).
prow(R,T) :- prow(R,T-1), pmove(2, T).

% hard restrictions (agent cannot leave window or move into a wall)
:- pcol(C,_), C = -radius - 1.
:- pcol(C,_), C = radius + 1.
:- prow(R,_), R = -radius - 1.
:- prow(R,_), R = radius + 1.
:- prow(R,T), pcol(C,T), wall(R,C).

% encode which action has been taken
action_taken(0, T) :- pcol(C,T), prow(R,T), prow(R+1,T+1).
action_taken(1, T) :- pcol(C,T), prow(R,T), pcol(C+1,T+1).
action_taken(2, T) :- pcol(C,T), prow(R,T), prow(R-1,T+1).
action_taken(3, T) :- pcol(C,T), prow(R,T), pcol(C-1,T+1).

action_taken(4, T) :- pcol(C,T), prow(R,T), pcol(C,T+1), prow(R,T+1).

% check the reward for each time step
reward(S, 0) :- action_taken(A,0), action(A,B), S = (horizon - 0) * B.


% movement of the frogs
fmove(F,0, T) | fmove(F,1, T)  | fmove(F,2, T) :- F=0..frogs-1, T=1..horizon - 1.
frow(F,R + 1,T) | frow(F,R - 1,T) :- frow(F,R,T-1), fmove(F,0, T).
fcol(F,C,T) :- fcol(F,C,T-1), fmove(F,0, T).
fcol(F,C,T) :- fcol(F,C,T-1), fmove(F,2, T).
fcol(F,C + 1,T) | fcol(F,C - 1,T) :- fcol(F,C,T-1), fmove(F,1, T).
frow(F,R,T) :- frow(F,R,T-1), fmove(F,1, T).
frow(F,R,T) :- frow(F,R,T-1), fmove(F,2, T).

% saturate if frogs make illegal move
sat :- fcol(_,C,_), C = -radius - 1.
sat :- fcol(_,C,_), C = radius + 1.
sat :- frow(_,R,_), R = -radius - 1.
sat :- frow(_,R,_), R = radius + 1.
sat :- frow(F,R,T), fcol(F,C,T), wall(R,C).

% check saturation criteria
ok(0).
ok(0,F) :- F=0..frogs-1.
sat :- ok(horizon-1).
:- not sat.

fcol(F,C,T) :- C = -radius..radius, sat, T=1..horizon-1, F=0..frogs-1.
frow(F,R,T) :- R = -radius..radius, sat, T=1..horizon-1, F=0..frogs-1.
fmove(F,0,T) :- F=0..frogs-1, T=1..horizon-1, sat.
fmove(F,1,T) :- F=0..frogs-1, T=1..horizon-1, sat.
fmove(F,2,T) :- F=0..frogs-1, T=1..horizon-1, sat.

% each frog either needs to be safely away from player or give penalty
ok(T,F) :- pcol(C,T), fcol(F,C',T), C != C', ok(T-1).
ok(T,F) :- prow(R,T), frow(F,R',T), R != R', ok(T-1).
good(T,G) | bad(T,G) :- G=0..frogs-1, T=1..horizon-1.
ok(T,F) :- bad(T,F).
ok(T,F) :- foutside(F), T=1..horizon-1.
ok(T) :- {ok(T,F) : F=0..frogs-1} = frogs, ok(T-1), frogs > 0.

% penalty for violating the norm of stepping onto a frog
penalty(P, T) :- P = (horizon - T) * 5, bad(T,F).

#minimize {S@2,T : penalty(S, T)}.
#maximize {R@1,T : reward(R,T)}.


#show action_taken/2.
%#show fcol/3.
%#show frow/3.
#show bad/2.
%#show pcol/2.
%#show prow/2.
