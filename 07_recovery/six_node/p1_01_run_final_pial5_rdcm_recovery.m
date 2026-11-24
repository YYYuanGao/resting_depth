function p1_01_run_final_pial5_rdcm_recovery(run_mode, n_reps, snrs, force)
% P1: study-specific recovery of the final pial5 EVC-cauPFC rDCM asymmetry.
%
% This program is intentionally independent of the historical
% bidirectional_dcm_recovery branch. It uses only:
%   * final outer-pial-5% six-node EVC-cauPFC inputs;
%   * final pial5 rDCM posterior A matrices from 15 participants;
%   * EVC = strict equal-ROI mean(V1,V2,V3) within depth;
%   * cauPFC = strict equal-ROI mean of 6d,6v,6mp,FEF,PEF,55b,SFL.
%
% Known-truth BOLD data are generated with the SPM biophysical fMRI model
% and inverted with the exact rDCM settings used for the empirical result.
% Three truth conditions are tested:
%   observed : final group-mean pial5 A matrix;
%   null     : the four prespecified FF/FB edges shifted so FB-FF = 0;
%   reversed : the same edges reflected so FB-FF = -observed(FB-FF).
%
% A matrix convention throughout: A(target,source).
%   FF = mean(A(4,3), A(5,3))
%   FB = mean(A(1,4), A(3,4))
%   H  = FB - FF
%
% Usage:
%   p1_01_run_final_pial5_rdcm_recovery('smoke', [], [], false)
%   p1_01_run_final_pial5_rdcm_recovery('pilot', [], [], false)
%   p1_01_run_final_pial5_rdcm_recovery('formal', [], [], false)
%   p1_01_run_final_pial5_rdcm_recovery('formal_n100', [], [], false)
%
% Defaults:
%   smoke : 1 repetition,  SNR = 3
%   pilot : 5 repetitions, SNR = 3
%   formal: 20 repetitions, SNR = [0.5 1 3]
%   formal_n100: 100 repetitions, SNR = [0.5 1 3]
%
% Existing successful case files are reused unless force=true. Failed or
% stale case files are rerun, making the program safely resumable.

SCRIPT_VERSION = '2026-08-27-final-pial5-rdcm-recovery-v3-n100';
DATA_ROOT = getenv('PROJECT_ROOT');
DERIV_ROOT = fullfile(DATA_ROOT, 'derivatives_0331');
SPM_PATH = getenv('SPM_PATH');
TAPAS_PATH = getenv('TAPAS_PATH');

INPUT_BASE = fullfile(DERIV_ROOT, 'S6_validation', ...
    'pial_depth_exclusion', 'matched_current_inputs', ...
    'pial5_current', 'cauPFC_EVC');
RDCM_BASE = fullfile(DERIV_ROOT, 'S6_validation', ...
    'pial_depth_exclusion', 'matched_current_rdcm', ...
    'pial5_current', 'cauPFC_EVC');
MEMBERSHIP_QC = fullfile(DERIV_ROOT, 'S6_validation', ...
    'pial_depth_exclusion', 'matched_current_inputs', ...
    'matched_group_layer_qc.csv');
OUT_BASE = fullfile(DERIV_ROOT, 'S6_validation', ...
    'p0_review_response', 'rdcm_fffb_recovery_pial5');

SUBJECTS = [1:14 16];
EXPECTED_LABELS = {'EVC_deep'; 'EVC_mid'; 'EVC_sup'; ...
    'cau_PFC_deep'; 'cau_PFC_mid'; 'cau_PFC_sup'};
BASE_SEED = 20260819;

if nargin < 1 || isempty(run_mode), run_mode = 'smoke'; end
run_mode = lower(char(string(run_mode)));
[default_reps, default_snrs] = defaults_for_mode(run_mode);
if nargin < 2 || isempty(n_reps), n_reps = default_reps; end
if nargin < 3 || isempty(snrs), snrs = default_snrs; end
if nargin < 4 || isempty(force), force = false; end

validateattributes(n_reps, {'numeric'}, {'scalar','integer','positive'});
validateattributes(snrs, {'numeric'}, {'vector','real','finite','positive'});
if numel(unique(snrs)) ~= numel(snrs)
    error('SNR values must be unique.');
end
if ~islogical(force) && ~(isnumeric(force) && isscalar(force))
    error('force must be a scalar logical.');
end
force = logical(force);

addpath(SPM_PATH);
addpath(genpath(fullfile(TAPAS_PATH, 'rDCM')));
spm('defaults', 'fMRI');
spm_jobman('initcfg');
required_functions = {'spm_dcm_fmri_priors', 'spm_int_J', ...
    'spm_fx_fmri', 'spm_gx_fmri', 'spm_unvec', 'spm_rand_mar', ...
    'spm_detrend', 'spm_Ce', ...
    'tapas_rdcm_estimate'};
assert_functions_available(required_functions);

run_dir = fullfile(OUT_BASE, run_mode);
case_dir = fullfile(run_dir, 'cases');
if ~exist(case_dir, 'dir'), mkdir(case_dir); end

fprintf('\n%s\nFINAL PIAL5 rDCM FF/FB RECOVERY: %s\n%s\n', ...
    repmat('=', 1, 88), upper(run_mode), repmat('=', 1, 88));
fprintf('Repetitions/cell: %d | SNR: %s | force=%d\n', ...
    n_reps, mat2str(snrs), force);

validate_membership_qc(MEMBERSHIP_QC, SUBJECTS);
[template, source_table, A_observed] = load_final_pial5_truth( ...
    INPUT_BASE, RDCM_BASE, SUBJECTS, EXPECTED_LABELS);
truths = make_truth_conditions(A_observed);
write_truth_artifacts(run_dir, source_table, truths, EXPECTED_LABELS);

n_timepoints = size(template.ts, 1);
n_nodes = size(template.ts, 2);
TR = double(template.TR);
TE = double(template.TE);
labels = normalize_labels(template.labels);
expected_cases = numel(truths) * numel(snrs) * n_reps;
fprintf('Final template: T=%d | nodes=%d | TR=%.4f | TE=%.4f\n', ...
    n_timepoints, n_nodes, TR, TE);
fprintf('Expected simulations/rDCM fits: %d\n', expected_cases);

all_rows = repmat(empty_result_row(), 0, 1);
for truth_index = 1:numel(truths)
    truth = truths(truth_index);
    for snr_index = 1:numel(snrs)
        SNR = double(snrs(snr_index));
        for rep_index = 1:n_reps
            % Common random numbers pair truth conditions within SNR/rep.
            seed = BASE_SEED + 10000 * snr_index + rep_index;
            case_id = sprintf('%s__snr-%g__rep-%03d', ...
                truth.name, SNR, rep_index);
            case_file = fullfile(case_dir, [case_id '.mat']);
            fprintf('\n[%s]\n', case_id);

            if exist(case_file, 'file') && ~force
                cached = load(case_file, 'script_version', 'case_row', ...
                    'truth', 'seed', 'SNR', 'rep_index', 'run_mode');
                if cache_is_current(cached, SCRIPT_VERSION, run_mode, ...
                        truth, SNR, rep_index, seed, n_timepoints, ...
                        n_nodes, TR, TE)
                    fprintf('  Reusing current successful cache.\n');
                    all_rows(end + 1, 1) = cached.case_row; %#ok<AGROW>
                    continue;
                end
                fprintf('  Existing case is stale or failed; rerunning.\n');
            end

            row = empty_result_row();
            row.case_id = string(case_id);
            row.run_mode = string(run_mode);
            row.truth_condition = string(truth.name);
            row.generator = "spm_biophysical";
            row.fit_method = "rDCM";
            row.repetition = rep_index;
            row.seed = seed;
            row.SNR = SNR;
            row.n_timepoints = n_timepoints;
            row.n_nodes = n_nodes;
            row.TR = TR;
            row.TE = TE;
            row.matrix_convention = "A(target,source)";
            row = add_true_metrics(row, truth.A_parameter);

            rng(seed, 'twister');
            innovations = spm_rand_mar(n_timepoints, n_nodes, 1/2) / 8;
            generation_timer = tic;
            generated_y = NaN(n_timepoints, n_nodes);
            generator_details = struct();
            try
                [generated_y, generator_details] = generate_spm_biophysical( ...
                    truth.A_parameter, innovations, n_timepoints, ...
                    n_nodes, TR, TE, SNR);
                row.generation_runtime_sec = toc(generation_timer);
                row.generation_success = true;
                row.generation_error = "";
            catch exception
                row.generation_runtime_sec = toc(generation_timer);
                row.generation_success = false;
                row.generation_error = string(getReport(exception, ...
                    'extended', 'hyperlinks', 'off'));
            end

            analysis_y = generated_y;
            scale_factor = NaN;
            rdcm_output = struct();
            rdcm_options = struct();
            A_estimated = NaN(n_nodes);
            fit_timer = [];
            if row.generation_success
                try
                    [analysis_y, scale_factor] = preprocess_like_empirical(generated_y);
                    if any(~isfinite(analysis_y), 'all') || ...
                            any(std(analysis_y, 0, 1) < 1e-10)
                        error('Generated/preprocessed data are non-finite or near-constant.');
                    end
                    fit_timer = tic;
                    [rdcm_output, rdcm_options] = fit_final_rdcm( ...
                        analysis_y, labels, TR, TE);
                    row.fit_runtime_sec = toc(fit_timer);
                    A_estimated = full(double(rdcm_output.Ep.A));
                    if ~isequal(size(A_estimated), [n_nodes n_nodes]) || ...
                            any(~isfinite(A_estimated), 'all')
                        error('rDCM returned a malformed or non-finite A matrix.');
                    end
                    row.fit_success = true;
                    row.fit_error = "";
                    row.scale_factor = scale_factor;
                    row = add_estimated_metrics(row, truth.A_parameter, A_estimated);
                catch exception
                    if ~isempty(fit_timer)
                        row.fit_runtime_sec = toc(fit_timer);
                    end
                    row.fit_success = false;
                    row.fit_error = string(getReport(exception, ...
                        'extended', 'hyperlinks', 'off'));
                end
            else
                row.fit_success = false;
                row.fit_error = "Generation failed: " + row.generation_error;
            end

            script_version = SCRIPT_VERSION;
            case_row = row;
            save(case_file, 'script_version', 'case_id', 'run_mode', ...
                'truth', 'SNR', 'rep_index', 'seed', 'innovations', ...
                'generated_y', 'analysis_y', 'scale_factor', ...
                'generator_details', 'rdcm_output', 'rdcm_options', ...
                'A_estimated', 'case_row', '-v7.3');
            all_rows(end + 1, 1) = row; %#ok<AGROW>
            write_current_results(all_rows, run_dir);

            fprintf('  generation=%s %.2fs | rDCM=%s %.2fs | true H=%+.4f | est H=%+.4f\n', ...
                pass_fail(row.generation_success), row.generation_runtime_sec, ...
                pass_fail(row.fit_success), row.fit_runtime_sec, ...
                row.true_H, row.estimated_H);
        end
    end
end

long_table = write_current_results(all_rows, run_dir);
write_run_manifest(run_dir, SCRIPT_VERSION, run_mode, n_reps, snrs, ...
    expected_cases, SUBJECTS, INPUT_BASE, RDCM_BASE, MEMBERSHIP_QC, ...
    template, truths, BASE_SEED);

n_generation_success = sum(long_table.generation_success);
n_fit_success = sum(long_table.fit_success);
fprintf('\n%s\nRECOVERY RUN COMPLETE\n%s\n', ...
    repmat('=', 1, 88), repmat('=', 1, 88));
fprintf('Generation success: %d/%d\n', n_generation_success, height(long_table));
fprintf('rDCM fit success:   %d/%d\n', n_fit_success, height(long_table));
fprintf('Long results: %s\n', fullfile(run_dir, 'rdcm_recovery_long.csv'));
fprintf('Next: run p1_02_analyze_final_pial5_rdcm_recovery.py\n');
if n_fit_success ~= expected_cases
    warning(['Recovery run contains failures. Do not interpret recovery ' ...
        'metrics until every expected case succeeds.']);
end
end


function current = cache_is_current(cached, script_version, run_mode, ...
        truth, SNR, repetition, seed, n_timepoints, n_nodes, TR, TE)
required = {'script_version','case_row','truth','seed','SNR', ...
    'rep_index','run_mode'};
current = all(isfield(cached, required));
if ~current, return; end
row = cached.case_row;
row_fields = {'generation_success','fit_success','truth_condition', ...
    'SNR','repetition','seed','n_timepoints','n_nodes','TR','TE'};
current = all(isfield(row, row_fields));
if ~current, return; end
current = strcmp(char(string(cached.script_version)), script_version) && ...
    strcmp(char(string(cached.run_mode)), run_mode) && ...
    strcmp(char(string(cached.truth.name)), char(string(truth.name))) && ...
    isequaln(double(cached.truth.A_parameter), double(truth.A_parameter)) && ...
    double(cached.SNR) == SNR && double(cached.rep_index) == repetition && ...
    double(cached.seed) == seed && ...
    strcmp(char(string(row.truth_condition)), char(string(truth.name))) && ...
    double(row.SNR) == SNR && double(row.repetition) == repetition && ...
    double(row.seed) == seed && double(row.n_timepoints) == n_timepoints && ...
    double(row.n_nodes) == n_nodes && abs(double(row.TR) - TR) < 1e-12 && ...
    abs(double(row.TE) - TE) < 1e-12 && ...
    logical(row.generation_success) && logical(row.fit_success);
end


function [n_reps, snrs] = defaults_for_mode(run_mode)
switch run_mode
    case 'smoke'
        n_reps = 1;
        snrs = 3;
    case 'pilot'
        n_reps = 5;
        snrs = 3;
    case 'formal'
        n_reps = 20;
        snrs = [0.5 1 3];
    case 'formal_n100'
        n_reps = 100;
        snrs = [0.5 1 3];
    otherwise
        error(['run_mode must be smoke, pilot, formal, or ' ...
            'formal_n100.']);
end
end


function assert_functions_available(names)
missing = {};
for index = 1:numel(names)
    if isempty(which(names{index}))
        missing{end + 1} = names{index}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    error('Missing required SPM/TAPAS functions: %s', strjoin(missing, ', '));
end
end


function validate_membership_qc(qc_file, subjects)
if ~exist(qc_file, 'file')
    error('Final membership QC file not found: %s', qc_file);
end
qc = readtable(qc_file, 'VariableNamingRule', 'preserve');
required = {'subject','branch','group','layer','members','n_members', ...
    'n_valid_members','strict_group'};
if ~all(ismember(required, qc.Properties.VariableNames))
    error('Membership QC lacks required columns: %s', qc_file);
end
branch = string(qc.('branch'));
group = string(qc.('group'));
members = string(qc.('members'));
subject = string(qc.('subject'));
layer = string(qc.('layer'));
expected_subjects = string(arrayfun(@(x) sprintf('sub-%02d', x), ...
    subjects, 'UniformOutput', false));

spec = {
    'EVC', 'V1,V2,V3', 3;
    'cau_PFC', '6d,6v,6mp,FEF,PEF,55b,SFL', 7
};
for index = 1:size(spec, 1)
    selected = branch == "pial5_current" & group == string(spec{index, 1});
    block = qc(selected, :);
    expected_rows = numel(subjects) * 3;
    if height(block) < expected_rows
        error('%s final pial5 QC expected at least %d rows; found %d.', ...
            spec{index, 1}, expected_rows, height(block));
    end
    if ~isequal(sort(unique(subject(selected))), sort(expected_subjects(:)))
        error('%s final pial5 QC participant set mismatch.', spec{index, 1});
    end
    keys = subject(selected) + "|" + layer(selected);
    [unique_keys, ~, key_index] = unique(keys);
    key_counts = accumarray(key_index, 1);
    if numel(unique_keys) ~= expected_rows || ...
            numel(unique(key_counts)) ~= 1 || ...
            ~isequal(sort(unique(layer(selected))), ...
            sort(["deep"; "mid"; "sup"]))
        error(['%s final pial5 QC subject/layer grid is incomplete or ' ...
            'has inconsistent duplicate counts.'], ...
            spec{index, 1});
    end
    if ~all(members(selected) == string(spec{index, 2}))
        error('%s final pial5 membership differs from %s.', ...
            spec{index, 1}, spec{index, 2});
    end
    if ~all(double(block.('n_members')) == spec{index, 3}) || ...
            ~all(double(block.('n_valid_members')) == spec{index, 3}) || ...
            ~all(table_column_to_logical(block.('strict_group')))
        error('%s final pial5 membership/validity audit failed.', spec{index, 1});
    end
    summary_fields = {'merged_mean','merged_std'};
    for field_index = 1:numel(summary_fields)
        field = summary_fields{field_index};
        if ismember(field, block.Properties.VariableNames)
            values = double(block.(field));
            if any(~isfinite(values))
                error('%s final pial5 QC %s contains non-finite values.', ...
                    spec{index, 1}, field);
            end
            for key_number = 1:numel(unique_keys)
                duplicate_values = values(key_index == key_number);
                if max(abs(duplicate_values - duplicate_values(1))) > 1e-12
                    error('%s final pial5 QC duplicate %s values conflict.', ...
                        spec{index, 1}, field);
                end
            end
        end
    end
    if ismember('merged_has_nonfinite', block.Properties.VariableNames) && ...
            any(table_column_to_logical(block.('merged_has_nonfinite')))
        error('%s final pial5 QC reports non-finite merged signals.', ...
            spec{index, 1});
    end
    fprintf('  %s QC: %d unique subject-layer rows; duplicate factor %d.\n', ...
        spec{index, 1}, numel(unique_keys), key_counts(1));
end
fprintf('Membership lineage: PASS (EVC=V1,V2,V3; cauPFC=7 frozen ROIs).\n');
end


function [template, source_table, A_group] = load_final_pial5_truth( ...
        input_base, rdcm_base, subjects, expected_labels)
template_file = fullfile(input_base, ...
    'sub-01_cauPFC_EVC_ts_merged_layered.mat');
if ~exist(template_file, 'file')
    error('Final pial5 template not found: %s', template_file);
end
template = load(template_file);
required = {'ts','labels','TR','TE','roi_groups','aggregation_rule'};
if ~all(isfield(template, required))
    error('Final pial5 template lacks required provenance fields: %s', template_file);
end
template.ts = double(template.ts);
labels = normalize_labels(template.labels);
if size(template.ts, 2) ~= 6 || ~isequal(labels(:), expected_labels(:))
    error('Unexpected final pial5 template nodes/order: %s', template_file);
end
if any(~isfinite(template.ts), 'all') || any(std(template.ts, 0, 1) < 1e-8)
    error('Final pial5 template contains invalid node time series.');
end
if ~contains(string(template.aggregation_rule), 'strict_equal_roi_mean')
    error('Final pial5 template is not strict equal-ROI composite-first.');
end
roi_text = string(template.roi_groups);
if ~(contains(roi_text, 'V1') && contains(roi_text, 'V2') && ...
        contains(roi_text, 'V3') && contains(roi_text, 'SFL'))
    error('Final pial5 template ROI provenance is incomplete.');
end

A_stack = NaN(6, 6, numel(subjects));
rows = repmat(struct('subject',"",'output_file',"",'input_file',"", ...
    'validation_condition',"",'network_name',"",'signed_FF',NaN, ...
    'signed_FB',NaN,'signed_H',NaN,'n_timepoints',NaN, ...
    'TR',NaN,'TE',NaN), numel(subjects), 1);
for index = 1:numel(subjects)
    subject = sprintf('sub-%02d', subjects(index));
    output_file = fullfile(rdcm_base, subject, 'rDCM_output.mat');
    if ~exist(output_file, 'file')
        error('Missing final pial5 rDCM output: %s', output_file);
    end
    loaded = load(output_file, 'output');
    if ~isfield(loaded, 'output') || ~isfield(loaded.output, 'Ep') || ...
            ~isfield(loaded.output.Ep, 'A')
        error('Final pial5 output lacks output.Ep.A: %s', output_file);
    end
    output = loaded.output;
    A = full(double(output.Ep.A));
    if ~isequal(size(A), [6 6]) || any(~isfinite(A), 'all')
        error('Invalid final pial5 A matrix: %s', output_file);
    end
    output_labels = normalize_labels(output.node_labels);
    if ~isequal(output_labels(:), expected_labels(:))
        error('Final pial5 output node order mismatch: %s', output_file);
    end
    if ~strcmp(char(string(output.validation_condition)), 'pial5_current') || ...
            ~strcmp(char(string(output.network_name)), 'cauPFC_EVC')
        error('Final output condition/network lineage mismatch: %s', output_file);
    end
    input_file = char(string(output.input_file));
    if ~contains(input_file, fullfile('matched_current_inputs', ...
            'pial5_current', 'cauPFC_EVC'))
        error('Final output input_file is not the matched pial5 branch: %s', output_file);
    end
    if ~exist(input_file, 'file')
        error('Final output points to a missing input: %s', input_file);
    end
    participant_input = load(input_file, 'ts', 'labels', 'TR', 'TE', ...
        'roi_groups', 'aggregation_rule', 'validation_branch');
    input_required = {'ts','labels','TR','TE','roi_groups', ...
        'aggregation_rule','validation_branch'};
    if ~all(isfield(participant_input, input_required))
        error('Final participant input lacks provenance fields: %s', input_file);
    end
    input_labels = normalize_labels(participant_input.labels);
    if ~isequal(size(participant_input.ts), size(template.ts)) || ...
            ~isequal(input_labels(:), expected_labels(:)) || ...
            any(~isfinite(participant_input.ts), 'all') || ...
            any(std(double(participant_input.ts), 0, 1) < 1e-8) || ...
            abs(double(participant_input.TR) - double(template.TR)) > 1e-12 || ...
            abs(double(participant_input.TE) - double(template.TE)) > 1e-12
        error('Final participant input acquisition/data mismatch: %s', input_file);
    end
    if ~strcmp(char(string(participant_input.validation_branch)), ...
            'pial5_current') || ...
            ~contains(string(participant_input.aggregation_rule), ...
            'strict_equal_roi_mean') || ...
            ~strcmp(char(string(participant_input.roi_groups)), ...
            char(string(template.roi_groups)))
        error('Final participant input composition lineage mismatch: %s', input_file);
    end

    A(1:7:end) = 0; % SPM one-state self-connection parameterization.
    A_stack(:, :, index) = A;
    m = hierarchy_metrics(A);
    rows(index).subject = string(subject);
    rows(index).output_file = string(output_file);
    rows(index).input_file = string(input_file);
    rows(index).validation_condition = string(output.validation_condition);
    rows(index).network_name = string(output.network_name);
    rows(index).signed_FF = m.FF;
    rows(index).signed_FB = m.FB;
    rows(index).signed_H = m.H;
    rows(index).n_timepoints = size(participant_input.ts, 1);
    rows(index).TR = double(participant_input.TR);
    rows(index).TE = double(participant_input.TE);
end
A_group = mean(A_stack, 3);
A_group(1:7:end) = 0;
source_table = struct2table(rows);
fprintf('Final truth lineage: PASS (%d pial5 EVC-cauPFC outputs).\n', numel(subjects));
group_metrics = hierarchy_metrics(A_group);
fprintf('Observed group truth: FF=%+.6f | FB=%+.6f | H=%+.6f\n', ...
    group_metrics.FF, group_metrics.FB, group_metrics.H);
end


function truths = make_truth_conditions(A_observed)
observed_metrics = hierarchy_metrics(A_observed);
H = observed_metrics.H;
if abs(H) < 1e-8
    error('Observed final pial5 truth has near-zero H; cannot construct reversal.');
end

A_null = A_observed;
A_null([sub2ind([6 6],4,3), sub2ind([6 6],5,3)]) = ...
    A_null([sub2ind([6 6],4,3), sub2ind([6 6],5,3)]) + H / 2;
A_null([sub2ind([6 6],1,4), sub2ind([6 6],3,4)]) = ...
    A_null([sub2ind([6 6],1,4), sub2ind([6 6],3,4)]) - H / 2;

A_reversed = A_observed;
A_reversed([sub2ind([6 6],4,3), sub2ind([6 6],5,3)]) = ...
    A_reversed([sub2ind([6 6],4,3), sub2ind([6 6],5,3)]) + H;
A_reversed([sub2ind([6 6],1,4), sub2ind([6 6],3,4)]) = ...
    A_reversed([sub2ind([6 6],1,4), sub2ind([6 6],3,4)]) - H;

names = {'observed','null','reversed'};
matrices = {A_observed, A_null, A_reversed};
truths = repmat(struct('name','','A_parameter',[],'A_effective',[], ...
    'FF',NaN,'FB',NaN,'H',NaN,'max_real_eigenvalue',NaN), 3, 1);
for index = 1:3
    A_parameter = matrices{index};
    A_parameter(1:7:end) = 0;
    A_effective = A_parameter - 0.5 * eye(6);
    max_eigenvalue = max(real(eig(A_effective)));
    if max_eigenvalue >= 0
        error('%s truth is dynamically unstable (max real eigenvalue %.6g).', ...
            names{index}, max_eigenvalue);
    end
    m = hierarchy_metrics(A_parameter);
    truths(index).name = names{index};
    truths(index).A_parameter = A_parameter;
    truths(index).A_effective = A_effective;
    truths(index).FF = m.FF;
    truths(index).FB = m.FB;
    truths(index).H = m.H;
    truths(index).max_real_eigenvalue = max_eigenvalue;
end
if abs(truths(2).H) > 1e-12 || ...
        abs(truths(3).H + truths(1).H) > 1e-12
    error('Null/reversed truth construction failed exact H checks.');
end
end


function write_truth_artifacts(run_dir, source_table, truths, labels)
if ~exist(run_dir, 'dir'), mkdir(run_dir); end
writetable(source_table, fullfile(run_dir, 'final_pial5_truth_sources.csv'));

summary_rows = repmat(struct('truth_condition',"",'true_FF',NaN, ...
    'true_FB',NaN,'true_H',NaN,'max_real_eigenvalue_effective_A',NaN, ...
    'stable',false), numel(truths), 1);
matrix_rows = repmat(struct('truth_condition',"",'target_index',NaN, ...
    'source_index',NaN,'target_label',"",'source_label',"", ...
    'A_parameter',NaN,'A_effective',NaN,'matrix_convention',"A(target,source)"), ...
    numel(truths) * 36, 1);
row_index = 0;
for truth_index = 1:numel(truths)
    t = truths(truth_index);
    summary_rows(truth_index).truth_condition = string(t.name);
    summary_rows(truth_index).true_FF = t.FF;
    summary_rows(truth_index).true_FB = t.FB;
    summary_rows(truth_index).true_H = t.H;
    summary_rows(truth_index).max_real_eigenvalue_effective_A = ...
        t.max_real_eigenvalue;
    summary_rows(truth_index).stable = t.max_real_eigenvalue < 0;
    for target = 1:6
        for source = 1:6
            row_index = row_index + 1;
            matrix_rows(row_index).truth_condition = string(t.name);
            matrix_rows(row_index).target_index = target;
            matrix_rows(row_index).source_index = source;
            matrix_rows(row_index).target_label = string(labels{target});
            matrix_rows(row_index).source_label = string(labels{source});
            matrix_rows(row_index).A_parameter = t.A_parameter(target, source);
            matrix_rows(row_index).A_effective = t.A_effective(target, source);
        end
    end
end
writetable(struct2table(summary_rows), ...
    fullfile(run_dir, 'truth_conditions.csv'));
writetable(struct2table(matrix_rows), ...
    fullfile(run_dir, 'truth_matrices_long.csv'));
save(fullfile(run_dir, 'truth_matrices.mat'), 'truths', 'source_table', '-v7.3');
end


function [y, details] = generate_spm_biophysical(A_parameter, innovations, ...
        n_timepoints, n_nodes, TR, TE, SNR)
options = struct('nonlinear',0,'two_state',0,'stochastic',0, ...
    'centre',1,'induced',1);
A_mask = ones(n_nodes);
B = zeros(n_nodes, n_nodes, 0);
C = eye(n_nodes);
D = zeros(n_nodes, n_nodes, 0);
pP = spm_dcm_fmri_priors(A_mask, B, C, D, options);
pP.A = A_parameter;
pP.B = B;
pP.C = C;
pP.D = D;
pP.transit = zeros(n_nodes, 1);
pP.decay = 0; % shared scalar in the one-state SPM fMRI model
pP.epsilon = 0;

U = struct();
U.u = repmat(innovations, 3, 1);
U.dt = TR;
M = struct();
M.f = 'spm_fx_fmri';
M.x = sparse(n_nodes, 5);
M.m = n_nodes;
M.l = n_nodes;
M.TE = TE;

x_all = spm_int_J(pP, M, U);
middle = (n_timepoints + 1):(2 * n_timepoints);
x = x_all(middle, :);
clean_y = zeros(n_timepoints, n_nodes);
for time_index = 1:n_timepoints
    state = spm_unvec(x(time_index, :), M.x);
    clean_y(time_index, :) = spm_gx_fmri(state, [], pP, M)';
end
signal_sd = std(clean_y, 0, 1);
if any(~isfinite(clean_y), 'all') || any(signal_sd < 1e-12)
    error('SPM biophysical generator produced invalid/near-zero signal.');
end
noise_z = randn(size(clean_y));
noise = noise_z * diag(signal_sd / SNR);
y = clean_y + noise;

details = struct();
details.generator = 'spm_biophysical';
details.noise_model = 'independent Gaussian observation noise per node';
details.innovation_model = 'MAR(1/2) endogenous innovations';
details.SNR_definition = 'node signal SD / node observation-noise SD';
details.signal_sd = signal_sd;
details.noise_sd_target = signal_sd / SNR;
details.pP = pP;
details.clean_y = clean_y;
details.x = x;
end


function [y, scale_factor] = preprocess_like_empirical(y)
y = spm_detrend(double(y));
data_range = max(y, [], 'all') - min(y, [], 'all');
scale_factor = 4 / max(data_range, 4);
y = y * scale_factor;
end


function [output, options] = fit_final_rdcm(y, labels, TR, TE)
n_timepoints = size(y, 1);
n_nodes = size(y, 2);
clear DCM
DCM.n = n_nodes;
DCM.v = n_timepoints;
DCM.Y.dt = TR;
DCM.Y.X0 = ones(n_timepoints, 1);
DCM.Y.Q = spm_Ce(repmat(n_timepoints, 1, n_nodes));
DCM.Y.y = y;
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

options = struct();
options.scale = 1;
options.estimateVL = 0;
options.u_shift = 0;
options.filter_str = 4;
options.compute_signal = 0;
options.wide_priors = 0;
[output, options] = tapas_rdcm_estimate(DCM, 'r', options, 1);
end


function row = add_true_metrics(row, A)
m = hierarchy_metrics(A);
row.true_A_4_3 = A(4,3);
row.true_A_5_3 = A(5,3);
row.true_A_1_4 = A(1,4);
row.true_A_3_4 = A(3,4);
row.true_FF = m.FF;
row.true_FB = m.FB;
row.true_H = m.H;
end


function row = add_estimated_metrics(row, A_true, A_estimated)
m = hierarchy_metrics(A_estimated);
row.estimated_A_4_3 = A_estimated(4,3);
row.estimated_A_5_3 = A_estimated(5,3);
row.estimated_A_1_4 = A_estimated(1,4);
row.estimated_A_3_4 = A_estimated(3,4);
row.estimated_FF = m.FF;
row.estimated_FB = m.FB;
row.estimated_H = m.H;
row.FF_bias = row.estimated_FF - row.true_FF;
row.FB_bias = row.estimated_FB - row.true_FB;
row.H_bias = row.estimated_H - row.true_H;
if abs(row.true_H) > 1e-10
    row.H_sign_recovered = sign(row.estimated_H) == sign(row.true_H);
else
    row.H_sign_recovered = NaN;
end
row.estimated_H_positive = row.estimated_H > 0;

mask = ~eye(size(A_true, 1));
truth_vector = A_true(mask);
estimate_vector = A_estimated(mask);
difference = estimate_vector - truth_vector;
row.offdiag_bias = mean(difference);
row.offdiag_rmse = sqrt(mean(difference .^ 2));
correlation = corrcoef(truth_vector, estimate_vector);
if isequal(size(correlation), [2 2])
    row.offdiag_pearson_r = correlation(1,2);
end
nonzero = abs(truth_vector) > 1e-8;
row.offdiag_sign_accuracy = mean( ...
    sign(estimate_vector(nonzero)) == sign(truth_vector(nonzero)));
true_key = [A_true(4,3), A_true(5,3), A_true(1,4), A_true(3,4)];
estimated_key = [A_estimated(4,3), A_estimated(5,3), ...
    A_estimated(1,4), A_estimated(3,4)];
row.key_edge_sign_accuracy = mean(sign(estimated_key) == sign(true_key));
end


function m = hierarchy_metrics(A)
m.FF = mean([A(4,3), A(5,3)]);
m.FB = mean([A(1,4), A(3,4)]);
m.H = m.FB - m.FF;
end


function long_table = write_current_results(rows, run_dir)
long_table = struct2table(rows);
writetable(long_table, fullfile(run_dir, 'rdcm_recovery_long.csv'));
end


function write_run_manifest(run_dir, version, run_mode, n_reps, snrs, ...
        expected_cases, subjects, input_base, rdcm_base, membership_qc, ...
        template, truths, base_seed)
manifest = struct();
manifest.analysis = 'Final pial5 EVC-cauPFC rDCM FF/FB recovery';
manifest.script_version = version;
manifest.run_mode = run_mode;
manifest.n_repetitions_per_cell = n_reps;
manifest.SNR = snrs;
manifest.expected_cases = expected_cases;
manifest.subjects_used_to_construct_truth = subjects;
manifest.n_truth_subjects = numel(subjects);
manifest.final_input_base = input_base;
manifest.final_rdcm_base = rdcm_base;
manifest.membership_qc = membership_qc;
manifest.n_timepoints = size(template.ts, 1);
manifest.n_nodes = size(template.ts, 2);
manifest.TR = double(template.TR);
manifest.TE = double(template.TE);
manifest.node_labels = normalize_labels(template.labels);
manifest.EVC_definition = 'strict equal-ROI mean of V1,V2,V3 within depth';
manifest.cauPFC_definition = ...
    'strict equal-ROI mean of 6d,6v,6mp,FEF,PEF,55b,SFL within depth';
manifest.matrix_convention = 'A(target,source)';
manifest.FF = 'mean(A(4,3),A(5,3))';
manifest.FB = 'mean(A(1,4),A(3,4))';
manifest.H = 'FB-FF';
manifest.truth_conditions = {truths.name};
manifest.generator = 'SPM one-state biophysical fMRI model';
manifest.noise_model = 'independent Gaussian observation noise per node';
manifest.SNR_definition = 'node signal SD / node observation-noise SD';
manifest.common_random_numbers = ...
    'truth conditions share seed within each SNR/repetition';
manifest.base_seed = base_seed;
manifest.fit = 'same TAPAS rDCM settings as final empirical pial5 model';
manifest.interpretation = ['Numerical recovery under specified simulation ' ...
    'assumptions; not evidence that the empirical effect is biologically true.'];
encoded = jsonencode(manifest);
file = fullfile(run_dir, 'rdcm_recovery_manifest.json');
handle = fopen(file, 'w');
if handle < 0, error('Cannot write manifest: %s', file); end
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '%s\n', encoded);
end


function row = empty_result_row()
row = struct( ...
    'case_id',"",'run_mode',"",'truth_condition',"", ...
    'generator',"",'fit_method',"",'repetition',NaN,'seed',NaN,'SNR',NaN, ...
    'n_timepoints',NaN,'n_nodes',NaN,'TR',NaN,'TE',NaN, ...
    'matrix_convention',"", ...
    'generation_success',false,'generation_runtime_sec',NaN, ...
    'generation_error',"",'fit_success',false,'fit_runtime_sec',NaN, ...
    'fit_error',"",'scale_factor',NaN, ...
    'true_A_4_3',NaN,'true_A_5_3',NaN,'true_A_1_4',NaN,'true_A_3_4',NaN, ...
    'estimated_A_4_3',NaN,'estimated_A_5_3',NaN, ...
    'estimated_A_1_4',NaN,'estimated_A_3_4',NaN, ...
    'true_FF',NaN,'true_FB',NaN,'true_H',NaN, ...
    'estimated_FF',NaN,'estimated_FB',NaN,'estimated_H',NaN, ...
    'FF_bias',NaN,'FB_bias',NaN,'H_bias',NaN, ...
    'H_sign_recovered',NaN,'estimated_H_positive',NaN, ...
    'offdiag_bias',NaN,'offdiag_rmse',NaN,'offdiag_pearson_r',NaN, ...
    'offdiag_sign_accuracy',NaN,'key_edge_sign_accuracy',NaN);
end


function labels = normalize_labels(labels_in)
if iscell(labels_in)
    labels = cellfun(@char, labels_in(:), 'UniformOutput', false);
elseif isstring(labels_in)
    labels = cellstr(labels_in(:));
elseif ischar(labels_in)
    labels = cellstr(labels_in);
else
    labels = cell(numel(labels_in), 1);
    for index = 1:numel(labels_in)
        labels{index} = char(labels_in(index));
    end
end
end


function values = table_column_to_logical(column)
if islogical(column)
    values = column;
elseif isnumeric(column)
    if any(~ismember(column, [0 1]))
        error('Logical QC column contains numeric values outside 0/1.');
    end
    values = logical(column);
else
    normalized = lower(strtrim(string(column)));
    valid = ismember(normalized, ["true","false","1","0"]);
    if ~all(valid)
        error('Logical QC column contains unrecognized values.');
    end
    values = normalized == "true" | normalized == "1";
end
values = values(:);
end


function value = pass_fail(success)
if success, value = 'PASS'; else, value = 'FAIL'; end
end
