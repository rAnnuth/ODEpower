function [inputVariables, stateVariables] = eval_input_states(inputs,states)
inputVariables = sym([]);
fn = fieldnames(inputs);
for k=numel(fn):-1:1
    inputVariables(k) = inputs.(fn{k});
    %inputNums(k) = inputs.(fn{k})(2);
    assignin('base', fn{k}, sym(fn{k}, 'real'))
end

stateVariables = sym([]);
fn = fieldnames(states);
for k=numel(fn):-1:1
    stateVariables(k) = states.(fn{k});
    % make them accessible for equations / maybe in function ws?
    assignin('base', fn{k}, sym(fn{k}, 'real'))
end
end