function update_block_parameters(model_name)
    warning('off', 'Simulink:Commands:SetParamLinkChangeWarn')
    % Define config mapping
    configs = {
        'piLine', @piLineHandler;
        'loadVarRL', @loadVarRLHandler;
        'VsourceR', @VsourceRHandler;
        'loadRL', @loadRLHandler;
        'loadR', @loadRHandler;
        'Buck_Switching', @buckHandler;
        'dabGAM', @dabGAMHandler;
        'DAB_Switching', @DABSwitchingHandler;
        'delayLpPIDroop', @delayLpPIDroopHandler;
        'delayPI', @delayPIHandler;
        'PI', @PIHandler
        'LpPI', @LpPIHandler;
    };

    % Get top-level blocks
    blocks = find_system(model_name, 'SearchDepth', 1, 'Type', 'Block');

    % Loop through blocks
    for i = 1:length(blocks)
        [block_name, ~] = extract(get_param(blocks{i}, 'Name'));
        for j = 1:size(configs, 1)
            if isequal(block_name, configs{j, 1})
                try
                    configs{j, 2}(model_name, get_param(blocks{i}, 'Name'));
                catch ME
                    fprintf('Failed for %s: %s\n', get_param(blocks{i}, 'Name'), ME.message);
                end
            end
        end
    end
    warning('on', 'Simulink:Commands:SetParamLinkChangeWarn')
end
function [s1,s2] = extract(str)
    idx = strfind(str, '_');        % Find indices of all underscores
    if isempty(idx)
        s1 = str;
        s2 = '';
    else
        lastUnderscore = idx(end);      % Get index of the last underscore
        s1 = extractBefore(str, lastUnderscore);        
        s2 = extractAfter(str, lastUnderscore);
    end
end

function piLineHandler(model_name, block_name)
    [~, num]= extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);
    set_param(path('Cline2'), 'c', sprintf('C_%s / 2 * Len_%s', num, num));
    set_param(path('Cline1'), 'c', sprintf('C_%s / 2 * Len_%s', num, num));
    set_param(path('Lline'), 'l', sprintf('L_%s * Len_%s', num, num));
    set_param(path('Rline'), 'r', sprintf('R_%s * Len_%s', num, num));
    set_param(path('Cline1_r'), 'r', sprintf('R_c_%s', num));
    set_param(path('Cline2_r'), 'r', sprintf('R_c_%s', num));
end

function loadVarRLHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);
    set_param(path('Lload'), 'l', sprintf('L_%s', num));
end

function VsourceRHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);
    set_param(path('esr'), 'r', sprintf('R_%s', num));
end

function loadRLHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);
    set_param(path('Lload'), 'l', sprintf('L_%s', num));
    set_param(path('Rload'), 'r', sprintf('R_%s', num));
end

function loadRHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);
    set_param(path('Rload'), 'r', sprintf('R_%s', num));
end

function buckHandler(model_name, block_name)
    parts = split(block_name, '_');
    if numel(parts) >= 3
        num = parts{3};
        path = @(comp) strcat(model_name, '/', block_name, '/', comp);
        set_param(path('Diode_1'), 'Ron', sprintf('V_d_%s', num));
        set_param(path('Diode_1'), 'Vf', sprintf('R_d_%s', num));
        set_param(path('Cout'), 'c', sprintf('C_%s', num));
        set_param(path('L'), 'l', sprintf('L_%s', num));
        set_param(path('L'), 'r', sprintf('R_l_%s', num));
        set_param(path('sw'), 'R_closed', sprintf('R_ds_%s', num));
        set_param(path('Cout_r'), 'r', sprintf('R_c_%s', num));
    else
        error('Block name format unexpected for Buck_Switching block: %s', block_name);
    end
end

function dabGAMHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);

    set_param(path('Trf'), 'N', sprintf('N_%s', num));

    for i = 1:8
        p = path(sprintf('S%d', i));
        set_param(p, 'Rds', sprintf('Rds_%s', num));
        set_param(p, 'Goff', sprintf('Goff_%s', num));
        set_param(p, 'diode_Goff', sprintf('Goff_%s', num));
        set_param(p, 'diode_Ron', sprintf('Rds_%s', num));
        set_param(p, 'diode_Vf', sprintf('Vf_%s', num));
    end

    set_param(path('Cin'), 'c', sprintf('Cin_%s', num));
    set_param(path('Lt'), 'l', sprintf('Lt_%s', num));
    set_param(path('Lt'), 'r', sprintf('Rt_%s', num));
    set_param(path('Cout'), 'MaskValueString', sprintf('Cout_%s|0', num));
    set_param(path('fs'), 'Value', sprintf('2*fs_%s', num));
    set_param(path('fs1'), 'Value', sprintf('2*fs_%s', num));
end

function delayPIHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);

    set_param(path('td'), 'gain', sprintf('1/Td_%s', num));
    set_param(path('td2'), 'gain', sprintf('1/Td_%s', num));
    set_param(path('kp'), 'gain', sprintf('Kp_%s', num));
    set_param(path('ki'), 'gain', sprintf('Ki_%s', num));
end

function delayLpPIDroopHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);

    set_param(path('fb3'), 'gain', sprintf('Fb_d_%s * 2 * pi', num));
    set_param(path('fb4'), 'gain', sprintf('Fb_d_%s * 2 * pi', num));
    set_param(path('Pref'), 'constant', sprintf('P_d_%s', num));
    set_param(path('K_d'), 'gain', sprintf('K_d_%s', num));
    set_param(path('td'), 'gain', sprintf('1/Td_%s', num));
    set_param(path('td2'), 'gain', sprintf('1/Td_%s', num));
    set_param(path('kp'), 'gain', sprintf('Kp_%s', num));
    set_param(path('ki'), 'gain', sprintf('Ki_%s', num));
    set_param(path('fb'), 'gain', sprintf('2*pi*Fb_%s', num));
    set_param(path('fb2'), 'gain', sprintf('2*pi*Fb_%s', num));
end

function LpPIHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);

    set_param(path('kp'), 'gain', sprintf('Kp_%s', num));
    set_param(path('ki'), 'gain', sprintf('Ki_%s', num));
    set_param(path('fb'), 'gain', sprintf('2*pi*Fb_%s', num));
    set_param(path('fb2'), 'gain', sprintf('2*pi*Fb_%s', num));
end

function PIHandler(model_name, block_name)
    [~, num] = extract(block_name);
    path = @(comp) strcat(model_name, '/', block_name, '/', comp);

    set_param(path('kp'), 'gain', sprintf('Kp_%s', num));
    set_param(path('ki'), 'gain', sprintf('Ki_%s', num));
end
