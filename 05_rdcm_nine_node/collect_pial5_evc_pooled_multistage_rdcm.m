function collect_pial5_evc_pooled_multistage_rdcm()
% Collect all nine-by-nine A matrices to an explicit long table.

DERIV = [getenv('PROJECT_ROOT') '/derivatives_0331'];
BASE = fullfile(DERIV, 'S6_validation', ...
    'pial5_multistage_bridge_evc_pooled');
RDCM_DIR = fullfile(BASE, 'rdcm');
OUT_DIR = fullfile(BASE, 'group');
SUBJECTS = [1:14, 16];
EXPECTED_LABELS = { ...
    'EVC_deep','EVC_mid','EVC_sup', ...
    'PPC_deep','PPC_mid','PPC_sup', ...
    'cauPFC_deep','cauPFC_mid','cauPFC_sup'};

if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end
edge_rows = cell(0, 15);
audit_rows = cell(0, 8);

for sub_num = SUBJECTS
    subject = sprintf('sub-%02d', sub_num);
    file = fullfile(RDCM_DIR, subject, 'rDCM_output.mat');
    if ~exist(file, 'file')
        audit_rows(end+1,:) = {subject,'MISSING','File not found', ...
            file,NaN,NaN,'',''}; %#ok<AGROW>
        continue;
    end
    try
        temp = load(file, 'output');
        if ~isfield(temp,'output') || ~isfield(temp.output,'Ep') || ...
                ~isfield(temp.output.Ep,'A')
            error('Missing output.Ep.A');
        end
        A = double(temp.output.Ep.A);
        if ~isequal(size(A), [9,9]) || any(~isfinite(A(:)))
            error('Expected a finite 9x9 A matrix');
        end
        labels = labels_to_cell(temp.output.node_labels);
        if ~isequal(labels(:)', EXPECTED_LABELS)
            error('Node labels/order do not match frozen definition');
        end
        scale_factor = get_numeric(temp.output, 'scale_factor');
        runtime = get_numeric(temp.output, 'time_rDCM');
        input_file = get_text(temp.output, 'input_file');
        for target = 1:9
            [target_stage, target_layer] = split_label(labels{target});
            for source = 1:9
                [source_stage, source_layer] = split_label(labels{source});
                if target == source
                    edge_class = 'self';
                elseif strcmp(target_stage, source_stage)
                    edge_class = 'within_stage_cross_layer';
                else
                    edge_class = 'between_stage';
                end
                edge_rows(end+1,:) = { ...
                    subject,target,source,labels{target},labels{source}, ...
                    target_stage,source_stage,target_layer,source_layer, ...
                    A(target,source),edge_class,'A(target,source)', ...
                    scale_factor,runtime,file}; %#ok<AGROW>
            end
        end
        audit_rows(end+1,:) = {subject,'PASS','',file,9,81,input_file, ...
            'outer-pial 5%; EVC=equal mean(V1,V2,V3)'}; %#ok<AGROW>
    catch err
        audit_rows(end+1,:) = {subject,'ERROR',err.message,file, ...
            NaN,NaN,'',''}; %#ok<AGROW>
    end
end

edge_names = {'subject','target_index','source_index','target_label', ...
    'source_label','target_stage','source_stage','target_layer', ...
    'source_layer','A','edge_class','matrix_convention','scale_factor', ...
    'time_rDCM_sec','output_file'};
audit_names = {'subject','status','message','output_file','n_nodes', ...
    'n_edges','input_file','input_semantics'};
edges = cell2table(edge_rows, 'VariableNames', edge_names);
audit = cell2table(audit_rows, 'VariableNames', audit_names);
edge_file = fullfile(OUT_DIR, ...
    'pial5_evc_pooled_multistage_A_edges_long.csv');
audit_file = fullfile(OUT_DIR, ...
    'pial5_evc_pooled_multistage_collection_audit.csv');
writetable(edges, edge_file);
writetable(audit, audit_file);

n_pass = sum(strcmp(audit.status, 'PASS'));
fprintf('\n%s\nEVC-pooled collection: PASS %d/%d\n', ...
    repmat('=',1,78), n_pass, numel(SUBJECTS));
disp(audit(:, {'subject','status','message'}));
fprintf('Edges: %s\n', edge_file);
if n_pass ~= numel(SUBJECTS) || height(edges) ~= numel(SUBJECTS)*81
    error(['Incomplete collection; inspect ' ...
        'pial5_evc_pooled_multistage_collection_audit.csv']);
end
fprintf('%s\n', repmat('=',1,78));
end


function [stage, layer] = split_label(label)
parts = strsplit(char(label), '_');
if numel(parts) < 2
    error('Invalid node label: %s', label);
end
layer = parts{end};
stage = strjoin(parts(1:end-1), '_');
if ~ismember(layer, {'deep','mid','sup'})
    error('Invalid layer suffix: %s', label);
end
end


function labels_cell = labels_to_cell(labels_in)
if iscell(labels_in)
    labels_cell = labels_in(:);
    for i = 1:numel(labels_cell)
        labels_cell{i} = char(labels_cell{i});
    end
elseif isstring(labels_in)
    labels_cell = cellstr(labels_in(:));
elseif ischar(labels_in)
    labels_cell = cellstr(labels_in);
else
    error('Unsupported labels');
end
end


function value = get_numeric(s, field_name)
if isfield(s, field_name) && isscalar(s.(field_name))
    value = double(s.(field_name));
else
    value = NaN;
end
end


function value = get_text(s, field_name)
if isfield(s, field_name)
    value = char(s.(field_name));
else
    value = '';
end
end
