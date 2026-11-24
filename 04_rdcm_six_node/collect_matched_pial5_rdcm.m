function collect_matched_pial5_rdcm()
% Extract subject-level hierarchy metrics and all A edges from matched rDCMs.
%
% Three conditions are kept separate:
%   baseline_current : current Step3 without added pial exclusion
%   pial5_current    : matched current Step3 with outer-pial 5% exclusion
%   old_primary      : historical S4 rDCM (pipeline replication reference)
%
% A(target,source) convention and preregistered hierarchy edges:
%   FF = mean(A(4,3), A(5,3))
%   FB = mean(A(1,4), A(3,4))
%   signed H   = FB - FF
%   absolute H = mean(abs(FB edges)) - mean(abs(FF edges))


DERIV_ROOT = [getenv('PROJECT_ROOT') '/derivatives_0331'];
MATCHED_BASE = fullfile(DERIV_ROOT, 'S6_validation', ...
    'pial_depth_exclusion', 'matched_current_rdcm');
OLD_BASE = fullfile(DERIV_ROOT, 'S4_rDCM_com_first', ...
    'layered_merged_cauPFC_EVC_tSNRclean');
OUT_DIR = fullfile(MATCHED_BASE, 'group_audit');

SUBJECTS = [1:14, 16];
NETWORKS = {'cauPFC_V1', 'cauPFC_EVC'};
CONDITIONS = {'baseline_current', 'pial5_current', 'old_primary'};

if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

metric_rows = cell(0, 19);
edge_rows = cell(0, 11);
audit_rows = cell(0, 6);

for ci = 1:numel(CONDITIONS)
    condition = CONDITIONS{ci};
    for ni = 1:numel(NETWORKS)
        network = NETWORKS{ni};
        for sub_num = SUBJECTS
            subject = sprintf('sub-%02d', sub_num);
            if strcmp(condition, 'old_primary')
                file = fullfile(OLD_BASE, network, subject, 'rDCM_output.mat');
            else
                file = fullfile(MATCHED_BASE, condition, network, subject, ...
                    'rDCM_output.mat');
            end

            if ~exist(file, 'file')
                audit_rows(end + 1, :) = {subject, network, condition, ...
                    'MISSING', 'File not found', file}; %#ok<AGROW>
                continue;
            end

            try
                tmp = load(file, 'output');
                if ~isfield(tmp, 'output') || ~isfield(tmp.output, 'Ep') || ...
                        ~isfield(tmp.output.Ep, 'A')
                    error('Missing output.Ep.A');
                end
                output = tmp.output;
                A = double(output.Ep.A);
                if ~isequal(size(A), [6, 6])
                    error('Expected 6x6 A, got %s', mat2str(size(A)));
                end
                if any(~isfinite(A(:)))
                    error('A contains non-finite values');
                end

                labels = get_labels(output, network);
                source_group = strip_layer_suffix(labels{1});
                ff_edges = [A(4, 3), A(5, 3)];
                fb_edges = [A(1, 4), A(3, 4)];
                signed_FF = mean(ff_edges);
                signed_FB = mean(fb_edges);
                signed_H = signed_FB - signed_FF;
                absolute_FF = mean(abs(ff_edges));
                absolute_FB = mean(abs(fb_edges));
                absolute_H = absolute_FB - absolute_FF;

                time_rdcm = get_numeric_field(output, 'time_rDCM');
                scale_factor = get_numeric_field(output, 'scale_factor');
                input_file = get_text_field(output, 'input_file');

                metric_rows(end + 1, :) = { ...
                    subject, network, condition, source_group, ...
                    signed_FF, signed_FB, signed_H, ...
                    absolute_FF, absolute_FB, absolute_H, ...
                    A(4, 3), A(5, 3), A(1, 4), A(3, 4), ...
                    time_rdcm, scale_factor, input_file, file, 6}; %#ok<AGROW>

                for target = 1:6
                    for source = 1:6
                        if target == source
                            edge_class = 'self';
                        elseif (target <= 3 && source >= 4) || ...
                                (target >= 4 && source <= 3)
                            edge_class = 'interregional';
                        else
                            edge_class = 'intraregional_cross_layer';
                        end
                        edge_rows(end + 1, :) = { ...
                            subject, network, condition, target, source, ...
                            labels{target}, labels{source}, A(target, source), ...
                            edge_class, file, 'A(target,source)'}; %#ok<AGROW>
                    end
                end

                audit_rows(end + 1, :) = {subject, network, condition, ...
                    'PASS', '', file}; %#ok<AGROW>
            catch err
                audit_rows(end + 1, :) = {subject, network, condition, ...
                    'ERROR', err.message, file}; %#ok<AGROW>
            end
        end
    end
end

metric_names = { ...
    'subject', 'network', 'condition', 'source_group', ...
    'signed_FF', 'signed_FB', 'signed_H', ...
    'absolute_FF', 'absolute_FB', 'absolute_H', ...
    'A_4_3_source_sup_to_cau_deep', ...
    'A_5_3_source_sup_to_cau_mid', ...
    'A_1_4_cau_deep_to_source_deep', ...
    'A_3_4_cau_deep_to_source_sup', ...
    'time_rDCM_sec', 'scale_factor', 'input_file', 'output_file', 'n_nodes'};
edge_names = {'subject', 'network', 'condition', 'target_index', ...
    'source_index', 'target_label', 'source_label', 'A', 'edge_class', ...
    'output_file', 'matrix_convention'};
audit_names = {'subject', 'network', 'condition', 'status', 'message', ...
    'output_file'};

metrics = cell2table(metric_rows, 'VariableNames', metric_names);
edges = cell2table(edge_rows, 'VariableNames', edge_names);
audit = cell2table(audit_rows, 'VariableNames', audit_names);

metric_file = fullfile(OUT_DIR, 'matched_pial5_subject_metrics.csv');
edge_file = fullfile(OUT_DIR, 'matched_pial5_A_edges_long.csv');
audit_file = fullfile(OUT_DIR, 'matched_pial5_collection_audit.csv');
writetable(metrics, metric_file);
writetable(edges, edge_file);
writetable(audit, audit_file);

fprintf('\n%s\nMatched pial5 rDCM collection audit\n', repmat('=', 1, 84));
summary = groupsummary(audit, {'condition', 'network', 'status'});
disp(summary(:, {'condition', 'network', 'status', 'GroupCount'}));
fprintf('Subject metrics: %s\n', metric_file);
fprintf('A edges:         %s\n', edge_file);
fprintf('Collection audit:%s\n', audit_file);

matched_pass = sum(strcmp(audit.condition, 'baseline_current') & ...
    strcmp(audit.status, 'PASS')) + ...
    sum(strcmp(audit.condition, 'pial5_current') & ...
    strcmp(audit.status, 'PASS'));
if matched_pass ~= 60
    error('Expected 60 matched PASS files, got %d. Inspect audit CSV.', ...
        matched_pass);
end
fprintf('Matched validation completeness: PASS 60/60\n');
fprintf('%s\n', repmat('=', 1, 84));
end


function labels = get_labels(output, network)
if isfield(output, 'node_labels')
    labels = labels_to_cell(output.node_labels);
else
    if strcmp(network, 'cauPFC_EVC')
        source = 'EVC';
    else
        source = 'V1';
    end
    labels = {[source '_deep']; [source '_mid']; [source '_sup']; ...
        'cau_PFC_deep'; 'cau_PFC_mid'; 'cau_PFC_sup'};
end
if numel(labels) ~= 6
    error('Expected six node labels, got %d', numel(labels));
end
labels = labels(:);
end


function value = get_numeric_field(s, field_name)
if isfield(s, field_name)
    value = double(s.(field_name));
    if ~isscalar(value)
        value = NaN;
    end
else
    value = NaN;
end
end


function value = get_text_field(s, field_name)
if isfield(s, field_name)
    value = char(s.(field_name));
else
    value = '';
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
    labels_cell = cell(numel(labels_in), 1);
    for i = 1:numel(labels_cell)
        labels_cell{i} = char(labels_in(i));
    end
end
labels_cell = labels_cell(:);
end


function base_label = strip_layer_suffix(label_in)
base_label = char(label_in);
base_label = strrep(base_label, '_deep', '');
base_label = strrep(base_label, '_mid', '');
base_label = strrep(base_label, '_sup', '');
end
