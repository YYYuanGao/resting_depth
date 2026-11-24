function rDCM_matched_current_baseline_pial5(subject, network_name, condition_name)
% Matched rDCM sensitivity: current baseline versus outer-pial 5% exclusion.
%
% The two branches use the same current extraction pipeline and identical
% strict equal-ROI composite construction.  Only pial-depth exclusion differs.
%
% Node order:
%   source_deep, source_mid, source_sup,
%   cau_PFC_deep, cau_PFC_mid, cau_PFC_sup
%
% Usage:
%   rDCM_matched_current_baseline_pial5([], [], [])
%   rDCM_matched_current_baseline_pial5(1, 'cauPFC_EVC', [])
%   rDCM_matched_current_baseline_pial5([], 'cauPFC_EVC', 'baseline_current')

DATA_ROOT = getenv('PROJECT_ROOT');
DERIV_ROOT = fullfile(DATA_ROOT, 'derivatives_0331');
TS_BASE = fullfile(DERIV_ROOT, 'S6_validation', 'pial_depth_exclusion', ...
    'matched_current_inputs');
OUTPUT_BASE = fullfile(DERIV_ROOT, 'S6_validation', 'pial_depth_exclusion', ...
    'matched_current_rdcm');

TAPAS_PATH = getenv('TAPAS_PATH');
SPM_PATH = getenv('SPM_PATH');

SUBJECTS = [1:14, 16];
NETWORKS = {'cauPFC_V1', 'cauPFC_EVC'};
CONDITIONS = {'baseline_current', 'pial5_current'};
LAYER_NAMES = {'deep', 'mid', 'sup'};
OUTPUT_NAME = 'rDCM_output.mat';

addpath(genpath(fullfile(TAPAS_PATH, 'rDCM')));
addpath(SPM_PATH);
spm('defaults', 'fMRI');
spm_jobman('initcfg');

if nargin < 1 || isempty(subject)
    subjects_analyze = SUBJECTS;
else
    subjects_analyze = subject;
end
if nargin < 2 || isempty(network_name)
    networks_analyze = NETWORKS;
else
    networks_analyze = normalize_list(network_name);
end
if nargin < 3 || isempty(condition_name)
    conditions_analyze = CONDITIONS;
else
    conditions_analyze = normalize_list(condition_name);
end

assert_members(networks_analyze, NETWORKS, 'network');
assert_members(conditions_analyze, CONDITIONS, 'condition');

for ci = 1:numel(conditions_analyze)
    condition = conditions_analyze{ci};
    for ni = 1:numel(networks_analyze)
        net = networks_analyze{ni};
        for sub_num = subjects_analyze
            Subject = sprintf('sub-%02d', sub_num);
            fprintf('\n%s\nMatched rDCM: %s | %s | %s\n%s\n', ...
                repmat('=', 1, 78), condition, net, Subject, ...
                repmat('=', 1, 78));

            ts_file = fullfile(TS_BASE, condition, net, ...
                sprintf('%s_%s_ts_merged_layered.mat', Subject, net));
            if ~exist(ts_file, 'file')
                fprintf('  [ERROR] Input not found: %s\n', ts_file);
                continue;
            end

            sub_out_dir = fullfile(OUTPUT_BASE, condition, net, Subject);
            out_file = fullfile(sub_out_dir, OUTPUT_NAME);
            if exist(out_file, 'file')
                fprintf('  Already done, skipping: %s\n', out_file);
                continue;
            end

            temp = load(ts_file);
            require_field(temp, 'ts', ts_file);
            require_field(temp, 'labels', ts_file);
            require_field(temp, 'TR', ts_file);
            require_field(temp, 'TE', ts_file);

            ts = double(temp.ts);
            labels = labels_to_cell(temp.labels);
            TR = double(temp.TR);
            TE = double(temp.TE);
            [n_timepoints, n_nodes] = size(ts);

            if n_nodes ~= 6 || numel(labels) ~= 6
                fprintf('  [ERROR] Expected six nodes/labels, got %d/%d.\n', ...
                    n_nodes, numel(labels));
                continue;
            end
            if any(~isfinite(ts(:)))
                fprintf('  [ERROR] Non-finite values detected. Skipping.\n');
                continue;
            end
            bad_std = find(std(ts, 0, 1) < 1e-8);
            if ~isempty(bad_std)
                fprintf('  [ERROR] Near-zero variance nodes detected. Skipping.\n');
                print_bad_nodes(labels, bad_std);
                continue;
            end

            % Keep every estimation choice identical to the primary rDCM.
            clear DCM output options
            DCM.n = n_nodes;
            DCM.v = n_timepoints;
            DCM.Y.dt = TR;
            DCM.Y.X0 = ones(DCM.v, 1);
            DCM.Y.Q = spm_Ce(repmat(DCM.v, 1, DCM.n));
            DCM.Y.y = ts;
            DCM.Y.name = labels(:)';

            DCM.U.u = zeros(n_timepoints * 16, 1);
            DCM.U.name = {'null'};
            DCM.U.dt = TR / 16;

            DCM.delays = repmat(TR / 2, DCM.n, 1);
            DCM.TE = TE;
            DCM.options.nonlinear = 0;
            DCM.options.two_state = 0;
            DCM.options.stochastic = 0;
            DCM.options.nograph = 1;
            DCM.options.wide_priors = 0;

            DCM.a = ones(DCM.n);
            DCM.b = zeros(DCM.n, DCM.n, 1);
            DCM.c = zeros(DCM.n, 1);
            DCM.d = zeros(DCM.n, DCM.n, 0);

            options.scale = 1;
            options.estimateVL = 0;
            options.u_shift = 0;
            options.filter_str = 4;
            options.compute_signal = 0;
            options.wide_priors = 0;

            DCM.Y.y = spm_detrend(DCM.Y.y);
            scale_factor = max(max(DCM.Y.y)) - min(min(DCM.Y.y));
            scale_factor = 4 / max(scale_factor, 4);
            DCM.Y.y = DCM.Y.y * scale_factor;

            fprintf('  Input: %s\n', ts_file);
            fprintf('  Data: %d timepoints x %d nodes | TR %.4f | TE %.4f\n', ...
                n_timepoints, n_nodes, TR, TE);
            fprintf('  Running rDCM...\n');
            timer = tic;
            try
                [output, options] = tapas_rdcm_estimate(DCM, 'r', options, 1);
            catch err
                fprintf('  [ERROR] rDCM failed: %s\n', err.message);
                continue;
            end
            output.time_rDCM = toc(timer);

            A = output.Ep.A;
            source_nodes = 1:3;
            target_nodes = 4:6;
            source_to_cau = A(target_nodes, source_nodes);
            cau_to_source = A(source_nodes, target_nodes);

            output.validation_condition = condition;
            output.network_name = net;
            output.node_labels = labels;
            output.source_group = strip_layer_suffix(labels{1});
            output.target_group = strip_layer_suffix(labels{4});
            output.layer_names = LAYER_NAMES;
            output.source_to_cau = source_to_cau;
            output.cau_to_source = cau_to_source;
            output.source_to_cau_mean = mean(source_to_cau(:), 'omitnan');
            output.cau_to_source_mean = mean(cau_to_source(:), 'omitnan');
            output.directional_diff = ...
                output.source_to_cau_mean - output.cau_to_source_mean;
            output.scale_factor = scale_factor;
            output.input_file = ts_file;
            output.matrix_convention = ...
                'A(target,source): directed influence from source to target';
            output.comparison_note = [ ...
                'Use pial5_current minus baseline_current as the pial-depth ' ...
                'sensitivity contrast. Historical primary outputs are a ' ...
                'separate pipeline-version replication reference.' ...
            ];

            if ~exist(sub_out_dir, 'dir')
                mkdir(sub_out_dir);
            end
            save(out_file, 'output', 'options', 'DCM', '-v7.3');
            fprintf('  Done %.1fs; saved: %s\n', output.time_rDCM, out_file);
        end
    end
end

fprintf('\n%s\nAll requested matched rDCMs complete.\nOutput: %s\n%s\n', ...
    repmat('=', 1, 78), OUTPUT_BASE, repmat('=', 1, 78));
end


function values = normalize_list(value)
if ischar(value) || isstring(value)
    values = cellstr(value);
elseif iscell(value)
    values = value;
else
    error('Expected a character vector, string, or cell array.');
end
values = values(:)';
end


function assert_members(values, allowed, label)
for i = 1:numel(values)
    if ~ismember(values{i}, allowed)
        error('Unknown %s: %s', label, values{i});
    end
end
end


function require_field(s, field_name, file_name)
if ~isfield(s, field_name)
    error('Input file lacks variable %s: %s', field_name, file_name);
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


function print_bad_nodes(labels, idx)
for k = 1:numel(idx)
    fprintf('    %s\n', labels{idx(k)});
end
end
