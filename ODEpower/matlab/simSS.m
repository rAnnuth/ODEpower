%% Linear Simulation (SS)
t_line = [tspan(1) : dt: tspan(2)];
y0 = zeros(length(StatesOperationPoint),1); % OP already included in SS

% Calculate the difference to OP input
u = inputModifier(inputNums,t_line) - repmat(inputNums',1,length(t_line));

%Simulate the linear model
[Y_line, t_line, X_line] = lsim(sys, u, t_line, y0);

