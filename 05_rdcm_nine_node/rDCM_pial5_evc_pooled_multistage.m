function rDCM_pial5_evc_pooled_multistage(subject)
% Estimate a fully connected nine-node rDCM from outer-pial-5% inputs.
%
% Stage definition:
%   EVC = equal ROI mean of V1, V2 and V3
%   EVC -> PPC -> cauPFC
%
% Node order:
%   EVC_deep/mid/sup, PPC_deep/mid/sup, cauPFC_deep/mid/sup.
%
% Usage:
%   rDCM_pial5_evc_pooled_multistage([])  % all 15 participants
%   rDCM_pial5_evc_pooled_multistage(1)

DERIV = [getenv('PROJECT_ROOT') '/derivatives_0331'];
BASE = fullfile(DERIV, 'S6_validation', 'pial5_multistage_bridge_evc_pooled');
TS_DIR = fullfile(BASE, 'timeseries');
OUT_DIR = fullfile(BASE, 'rdcm');
TAPAS_PATH = getenv('TAPAS_PATH');
SPM_PATH = getenv('SPM_PATH');

SUBJECTS = [1:14, 16];
EXPECTED_LABELS = { ...
    'EVC_deep','EVC_mid','EVC_sup', ...
    'PPC_deep','PPC_mid','PPC_sup', ...
    'cauPFC_deep','cauPFC_mid','cauPFC_sup'};
STAGES = {'EVC','PPC','cauPFC'};
LAYERS = {'deep','mid','sup'};

addpath(genpath(fullfile(TAPAS_PATH, 'rDCM')));
addpath(SPM_PATH);
spm('defaults', 'fMRI');
spm_jobman('initcfg');

if nargin < 1 || isempty(subject)
    subjects_analyze = SUBJECTS;
else
    subjects_analyze = subject;
end

for sub_num = subjects_analyze
    Subject = sprintf('sub-%02d', sub_num);
    input_file = fullfile(TS_DIR, ...
        sprintf('%s_pial5_evc_pooled_multistage_layered.mat', Subject));
    sub_out = fullfile(OUT_DIR, Subject);
    output_file = fullfile(sub_out, 'rDCM_output.mat');
    fprintf('\n%s\nEVC-pooled three-stage pial5 rDCM | %s\n%s\n', ...
        repmat('=',1,78), Subject, repmat('=',1,78));

    if ~exist(input_file, 'file')
        fprintf('[ERROR] Missing input: %s\n', input_file);
        continue;
    end
    if exist(output_file, 'file')
        fprintf('Already done, skipping: %s\n', output_file);
        continue;
    end

    temp = load(input_file);
    require_field(temp, 'ts', input_file);
    require_field(temp, 'labels', input_file);
    require_field(temp, 'TR', input_file);
    require_field(temp, 'TE', input_file);
    ts = double(temp.ts);
    labels = labels_to_cell(temp.labels);
    TR = double(temp.TR);
    TE = double(temp.TE);
    [n_timepoints, n_nodes] = size(ts);

    if n_nodes ~= 9 || numel(labels) ~= 9
        error('%s: expected 9 nodes/labels, got %d/%d', ...
            Subject, n_nodes, numel(labels));
    end
    if ~isequal(labels(:)', EXPECTED_LABELS)
        error('%s: node labels/order differ from frozen definition', Subject);
    end
    if any(~isfinite(ts(:)))
        error('%s: input has non-finite values', Subject);
    end
    if any(std(ts,0,1) < 1e-8)
        error('%s: input has a near-zero variance node', Subject);
    end

    clear DCM output options
    DCM.n = n_nodes;
    DCM.v = n_timepoints;
    DCM.Y.dt = TR;
    DCM.Y.X0 = ones(n_timepoints, 1);
    DCM.Y.Q = spm_Ce(repmat(n_timepoints, 1, n_nodes));
    DCM.Y.y = ts;
    DCM.Y.name = labels(:)';
    DCM.U.u = zeros(n_timepoints * 16, 1);
    DCM.U.name = {'null'};
    DCM.U.dt = TR / 16;
    DCM.delays = repmat(TR / 2, n_nodes, 1);
    DCM.TE = TE;
    DCM.options.nonlinear = 0;
    DCM.options.two_state = 0;
    DCM.options.stochastic = 0;
    DCM.options.nograph = 1;
    DCM.options.wide_priors = 0;
    DCM.a = ones(n_nodes);
    DCM.b = zeros(n_nodes, n_nodes, 1);
    DCM.c = zeros(n_nodes, 1);
    DCM.d = zeros(n_nodes, n_nodes, 0);

    options.scale = 1;
    options.estimateVL = 0;
    options.u_shift = 0;
    options.filter_str = 4;
    options.compute_signal = 0;
    options.wide_priors = 0;

    % Match the existing endpoint and four-stage rDCM preprocessing.
    DCM.Y.y = spm_detrend(DCM.Y.y);
    scale_factor = max(DCM.Y.y(:)) - min(DCM.Y.y(:));
    scale_factor = 4 / max(scale_factor, 4);
    DCM.Y.y = DCM.Y.y * scale_factor;

    fprintf('Input: %s\n', input_file);
    fprintf(['Data: %d timepoints x %d nodes | TR %.4f | TE %.4f | ' ...
        'scale %.6f\n'], n_timepoints, n_nodes, TR, TE, scale_factor);
    timer = tic;
    try
        [output, options] = tapas_rdcm_estimate(DCM, 'r', options, 1);
    catch err
        fprintf('[ERROR] rDCM failed for %s: %s\n', Subject, err.message);
        continue;
    end
    output.time_rDCM = toc(timer);
    output.node_labels = labels;
    output.stage_names = STAGES;
    output.layer_names = LAYERS;
    output.scale_factor = scale_factor;
    output.input_file = input_file;
    output.outer_pial_exclusion = 'outer-pial 5% final input';
    output.EVC_definition = 'equal ROI mean of V1, V2 and V3';
    output.matrix_convention = ...
        'A(target,source): directed influence from source to target';
    output.model_note = [ ...
        'Three frozen, non-overlapping stages; EVC is V1/V2/V3 pooled; ' ...
        'three layers per stage; fully connected intrinsic A matrix; ' ...
        'resting-state null input.'];

    if ~exist(sub_out, 'dir')
        mkdir(sub_out);
    end
    save(output_file, 'output', 'options', 'DCM', '-v7.3');
    fprintf('Done %.2fs; saved: %s\n', output.time_rDCM, output_file);
end

fprintf('\nAll requested EVC-pooled rDCMs complete. Output: %s\n', OUT_DIR);
end


function require_field(s, field_name, file_name)
if ~isfield(s, field_name)
    error('Input lacks %s: %s', field_name, file_name);
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
    error('Unsupported label representation');
end
labels_cell = labels_cell(:);
end
