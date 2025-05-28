%% Component Parameters
outputVariables = stateVariables; % We want to observe state variables

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Do not change!

% Initialize an empty symbolic array for state derivatives
stateDerivatives = sym('a',[length(stateVariables),1]);
% Field names of the derivatives struct
fields = fieldnames(pyodes);
% Iterate over each field in the struct and construct the state derivatives array
for i = length(fields):-1:1
    stateDerivatives(i) = pyodes.(fields{i});
end