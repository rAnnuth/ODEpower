t_lin = [tspan(1) : dt: tspan(2)]; % SS simulation flips time vector
% The input is set for all inputVariables
paths = {};
for i = length(inputVariables) : -1 : 1
    paths(i) = {char(inputVariables(i))};
end

for i = 1: length(inputVariables) % Input variables
    simIn = setVariable(simIn,char(inputVariables(i)),inputNums(i));
end
for i = 1: length(params) % Parameters
    simIn = setVariable(simIn,params{i}{1}, params{i}{2});
end

% Create Series object for step data
ds = Simulink.SimulationData.Dataset;
ds{1} = struct;
u = inputModifier(inputNums',t_lin);
il = size(u);
for i = 1 : il(1)
    series = timeseries(u(i,:),t_lin);
    ds{1}.(paths{i}) = series;    
end

simIn = simIn.setModelParameter('StopTime',num2str(tspan(2)));
simIn = simIn.setExternalInput(ds);
out = sim(simIn); % simulate