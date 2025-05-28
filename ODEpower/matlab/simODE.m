
% genericOdefun within ode45
if enableLim
    [t_nlin, X_nlin] = ode23t(@(t, y) genericOdefun(t, y, lim_stateDerivatives, stateVariables, inputVariables, inputNums, inputModifier), tspan, y0);
else
    [t_nlin, X_nlin] = ode23t(@(t, y) genericOdefun(t, y, stateDerivatives, stateVariables, inputVariables, inputNums, inputModifier), tspan, y0);
end

% Calculate the input vector
u_nlin = inputModifier(inputNums,t_nlin');

%%
function dydt = genericOdefun(t, y, stateDerivatives, stateVariables, inputVariables, inputNums, inputModifier)
% Modify input
inputNums = inputModifier(inputNums,t);

% Substitute the current state and input values into the derivatives
dydt = double(subs(stateDerivatives, [stateVariables, inputVariables]', [y; inputNums(:)]));
end