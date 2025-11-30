%%%% INPUT %%%%

% constants
#const size=15.
#const horizon=3.

% constant atoms
wall(1, 10).                % wall(COLUMN, ROW)
grass(2, 9, 1).          % grass(COLUMN, ROW, ID)
lake(3,10, 1).              % lake(COLUMN, ROW, ID)


% variable atoms
agent(3, 10, 0).           % agent(COLUMN, ROW, TIME_STEP)
grass_timer(1, 30, 0).        % grass_timer(ID, COUNTER, TIME_STEP)
lake_timer(1, 4, 0).        % lake_timer(ID, COUNTER, TIME_STEP)


% world atoms
frog(1,2,0,0,239).            % frog(COLUMN, ROW, ID, TIME_STEP, WORLD)


































