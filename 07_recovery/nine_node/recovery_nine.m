function recovery_nine(config_file, mode)
% Nine-node recovery; original MRI, inputs, fits and ROI-size runs are read-only.
% Run preflight then pilot; do not interpret technical completion as recovery success.
if nargin<2, mode='preflight'; end
mode=char(mode); selftest();
if strcmp(mode,'selftest'), fprintf('Self-test PASS\n'); return; end
assert(ismember(mode,{'preflight','smoke','pilot','formal'}),'Unknown mode');
c=jsondecode(fileread(config_file)); validate_config(c);
out=fullfile(c.project_root,c.output_relative);
assert(~isfile(out),'Output is a file');
if ~isfolder(out), mkdir(out); end
report_dir=fullfile(out,mode); if ~isfolder(report_dir), mkdir(report_dir); end
try
    addpath(c.spm_path); addpath(genpath(fullfile(c.tapas_path,'rDCM')));
    spm('defaults','fMRI'); spm_jobman('initcfg');
    required={'spm_dcm_fmri_priors','spm_int_J','spm_fx_fmri','spm_gx_fmri', ...
        'spm_unvec','spm_rand_mar','spm_detrend','spm_Ce','tapas_rdcm_estimate'};
    for i=1:numel(required), assert(~isempty(which(required{i})),['Missing ' required{i}]); end
    [A,sources]=load_truth(c);
    truths=make_truths(A);
    runtime=struct('matlab',version,'files',struct());
    for i=1:numel(required)
        f=which(required{i}); runtime.files.(required{i})=struct('path',f,'sha256',sha(f));
    end
    code=fullfile(c.project_root,c.empirical_code_relative);
    assert(strcmp(sha(code),c.empirical_code_sha256),'Empirical source hash differs; review before running');
    lock=struct('config',c,'driver_sha256',sha([mfilename('fullpath') '.m']), ...
        'empirical_source_sha256',sha(code),'runtime',runtime,'sources',sources);
    token=sha_text(jsonencode(lock)); lock.token=token;
    lock_file=fullfile(out,'run_lock.json');
    if isfile(lock_file)
        prior=jsondecode(fileread(lock_file));
        assert(strcmp(prior.token,token),'Run inputs/config/code/runtime changed: use a NEW output directory');
    else
        write_json(lock_file,lock);
    end
    writetable(records_table(sources),fullfile(out,'source_checks.csv'));
    write_truth(out,truths,c.labels);
    write_json(fullfile(out,'preflight.json'),struct('status','PASS','token',token, ...
        'n_subjects',numel(sources),'n_nodes',9,'n_timepoints',c.n_timepoints, ...
        'scope','Input/label/saved-signal/source/stability checks only; no recovery fits'));
    if strcmp(mode,'preflight'), fprintf('Preflight PASS: %s\n',out); return; end
    if strcmp(mode,'formal')
        f=fullfile(out,'pilot','status.json'); assert(isfile(f),'Run and review pilot before formal');
        p=jsondecode(fileread(f));
        assert(strcmp(p.status,'COMPLETE') && strcmp(p.token,token),'Pilot technical completion/token required');
    end
    if strcmp(mode,'smoke'), snrs=max(c.snrs); reps=1;
    elseif strcmp(mode,'pilot'), snrs=c.snrs; reps=c.pilot_repeats;
    else, snrs=c.snrs; reps=c.formal_repeats; end
    case_dir=fullfile(report_dir,'cases'); if ~isfolder(case_dir), mkdir(case_dir); end
    rows={}; started=tic;
    for ti=1:3
        truth=truths(ti);
        for si=1:numel(snrs)
            snr=snrs(si); canonical_si=find(c.snrs==snr);
            for ri=1:reps
                seed=c.master_seed+10000*canonical_si+ri;
                id=sprintf('%s_snr-%g_rep-%03d',truth.name,snr,ri);
                f=fullfile(case_dir,[id '.mat']);
                if isfile(f)
                    saved=load(f,'row','token');
                    assert(strcmp(saved.token,token),'Case token mismatch');
                    row=saved.row; % Failures are retained too, never selectively discarded.
                else
                    row=struct('case_id',id,'truth',truth.name,'SNR',snr,'repetition',ri, ...
                        'seed',seed,'success',false,'generation_seconds',NaN,'fit_seconds',NaN, ...
                        'scale_factor',NaN,'last_warning','','last_warning_id','','error','');
                    log_file=fullfile(case_dir,[id '.log']);
                    diary(log_file); diary_cleanup=onCleanup(@() diary('off')); lastwarn('');
                    timer=tic; A_estimated=NaN(9); rdcm_output=struct(); rdcm_options=struct();
                    generated_y=[]; analysis_y=[]; details=struct(); innovations=[];
                    phase='generate';
                    try
                        rng(seed,'twister');
                        innovations=spm_rand_mar(c.n_timepoints,9,1/2)/8;
                        [generated_y,details]=generate_spm_biophysical(truth.A,innovations, ...
                            c.n_timepoints,9,c.TR,c.TE,snr);
                        row.generation_seconds=toc(timer);
                        [analysis_y,row.scale_factor]=preprocess_like_empirical(generated_y);
                        assert(all(isfinite(analysis_y),'all') && all(std(analysis_y)>1e-10),'Invalid simulated signal');
                        phase='fit'; timer=tic;
                        [rdcm_output,rdcm_options]=fit_final_rdcm(analysis_y,cellstr(string(c.labels)),c.TR,c.TE);
                        row.fit_seconds=toc(timer); A_estimated=full(double(rdcm_output.Ep.A));
                        assert(isequal(size(A_estimated),[9 9]) && all(isfinite(A_estimated),'all'),'Invalid recovered A');
                        row.success=true;
                    catch err
                        if strcmp(phase,'generate'), row.generation_seconds=toc(timer);
                        else, row.fit_seconds=toc(timer); end
                        row.error=getReport(err,'extended','hyperlinks','off');
                        fprintf(2,'%s\n',row.error);
                    end
                    [row.last_warning,row.last_warning_id]=lastwarn;
                    row=metrics(row,truth.A,A_estimated);
                    clear diary_cleanup
                    temp=[f '.partial.mat'];
                    save(temp,'row','token','truth','innovations','generated_y','analysis_y', ...
                        'details','rdcm_output','rdcm_options','A_estimated','-v7.3');
                    movefile(temp,f); % Commit checkpoint only after complete save.
                end
                rows{end+1,1}=row; %#ok<AGROW>
                writetable(records_table(vertcat(rows{:})),fullfile(report_dir,'recovery_cases.csv'));
            end
        end
    end
    tbl=records_table(vertcat(rows{:})); expected=3*numel(snrs)*reps;
    status='COMPLETE'; if sum(tbl.success)~=expected, status='INCOMPLETE'; end
    write_json(fullfile(report_dir,'status.json'),struct('status',status,'token',token, ...
        'expected_cases',expected,'successful',sum(tbl.success), ...
        'cases_with_last_warning',sum(strlength(string(tbl.last_warning))>0), ...
        'invocation_seconds',toc(started),'scope','Technical completion; inspect recovery errors/bias and ALL logs'));
    fprintf('%s: %d/%d fits. Reports: %s\n',status,sum(tbl.success),expected,report_dir);
catch err
    write_json(fullfile(report_dir,'error_status.json'),struct('status','ERROR', ...
        'error',getReport(err,'extended','hyperlinks','off')));
    rethrow(err);
end
end

function validate_config(c)
assert(numel(c.subjects)==15 && numel(unique(string(c.subjects)))==15,'Expected 15 unique participants');
expected=compose('sub-%02d',[1:14 16]); assert(isequal(string(c.subjects(:)),expected(:)),'Participant set/order mismatch');
labels={'EVC_deep';'EVC_mid';'EVC_sup';'PPC_deep';'PPC_mid';'PPC_sup';'cauPFC_deep';'cauPFC_mid';'cauPFC_sup'};
assert(isequal(string(c.labels(:)),string(labels)),'Frozen node order mismatch');
assert(c.n_timepoints==218 && abs(c.TR-2.2)<1e-12 && abs(c.TE-.026)<1e-12,'Acquisition differs');
for f={'pilot_repeats','formal_repeats','master_seed'}
    validateattributes(c.(f{1}),{'numeric'},{'scalar','positive','integer'});
end
assert(c.pilot_repeats<10000 && c.formal_repeats<10000,'Seed spacing exceeded');
validateattributes(c.snrs,{'numeric'},{'vector','real','finite','positive'});
assert(numel(unique(c.snrs))==numel(c.snrs),'Duplicate SNR');
validateattributes(c.saved_signal_atol,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(c.saved_signal_rtol,{'numeric'},{'scalar','finite','nonnegative'});
% New output must be inside a dedicated recovery branch, never an original input branch.
assert(startsWith(c.output_relative,'S6_validation/rdcm_nine_recovery/'),'Use dedicated recovery output branch');
assert(~contains(c.output_relative,'..'),'Output traversal is forbidden');
end

function [A,rows]=load_truth(c)
stack=zeros(9,9,15); source_rows=cell(15,1);
for i=1:15
    s=char(c.subjects{i});
    input=fullfile(c.project_root,strrep(c.input_relative,'{subject}',s));
    fit=fullfile(c.project_root,strrep(c.fit_relative,'{subject}',s));
    x=load(input,'ts','labels','TR','TE'); y=load(fit,'output','DCM','options');
    assert(all(isfield(x,{'ts','labels','TR','TE'})),'Input fields missing');
    assert(all(isfield(y,{'output','DCM','options'})),'Saved fit fields missing');
    assert(isequal(string(x.labels(:)),string(c.labels(:))),'Input node order mismatch');
    assert(isequal(string(y.output.node_labels(:)),string(c.labels(:))),'Output node order mismatch');
    assert(isequal(string(y.DCM.Y.name(:)),string(c.labels(:))),'DCM node order mismatch');
    assert(strcmp(char(y.output.input_file),input),'Saved fit points to a different input');
    assert(isequal(size(x.ts),[c.n_timepoints 9]) && all(isfinite(x.ts),'all') && all(std(double(x.ts))>1e-8),'Invalid input signal');
    assert(abs(double(x.TR)-c.TR)<1e-12 && abs(double(x.TE)-c.TE)<1e-12,'Acquisition mismatch');
    [signal,scale]=preprocess_like_empirical(x.ts);
    saved=double(y.DCM.Y.y);
    assert(isequal(size(saved),size(signal)) && all(isfinite(saved),'all'),'Invalid saved DCM signal');
    diff=abs(signal-saved);
    assert(all(diff<=c.saved_signal_atol+c.saved_signal_rtol*abs(saved),'all'),'Saved DCM signal mismatch');
    assert(y.DCM.n==9 && y.DCM.v==c.n_timepoints && abs(y.DCM.Y.dt-c.TR)<1e-12 && abs(y.DCM.TE-c.TE)<1e-12,'Saved DCM settings mismatch');
    assert(isequal(size(y.DCM.a),[9 9]) && all(y.DCM.a==1,'all') && all(y.DCM.b==0,'all') && all(y.DCM.c==0,'all') && all(y.DCM.U.u==0,'all'),'Connectivity/input settings mismatch');
    for f={'nonlinear','two_state','stochastic','wide_priors'}
        assert(y.DCM.options.(f{1})==0,['Saved DCM option mismatch: ' f{1}]);
    end
    fields={'scale','estimateVL','u_shift','filter_str','compute_signal','wide_priors'}; values=[1 0 0 4 0 0]; absent={};
    for j=1:numel(fields)
        name=fields{j};
        if isfield(y.options,name), assert(isequal(double(y.options.(name)),values(j)),['Saved option differs: ' name]);
        else
            assert(ismember(name,{'scale','estimateVL','wide_priors'}),['Unexpected missing option: ' name]);
            absent{end+1}=name; %#ok<AGROW> % Known output omission, source evidence only.
        end
    end
    ai=full(double(y.output.Ep.A));
    assert(isequal(size(ai),[9 9]) && all(isfinite(ai),'all'),'Invalid original A');
    original_diagonal=diag(ai); ai(1:10:end)=0; stack(:,:,i)=ai;
    m=path_metrics(ai,1,3);
    row=struct('subject',s,'input_file',input,'input_sha256',sha(input), ...
        'fit_file',fit,'fit_sha256',sha(fit),'saved_signal_max_abs_error',max(diff,[],'all'), ...
        'scale_factor',scale,'FF',m(1),'FB',m(2),'H',m(3), ...
        'original_diagonal_min',min(original_diagonal),'original_diagonal_max',max(original_diagonal), ...
        'unretained_options_source_evidence_only',strjoin(absent,';'));
    source_rows{i}=row;
end
rows=vertcat(source_rows{:});
A=mean(stack,3); A(1:10:end)=0;
end

function truths=make_truths(A)
assert(isequal(size(A),[9 9]) && all(isfinite(A),'all'),'Nine-node truth required');
A(1:10:end)=0; h=path_metrics(A,1,3); h=h(3);
assert(abs(h)>1e-8,'Cannot form nonzero reversal');
ff=sub2ind([9 9],[7 8],[3 3]); fb=sub2ind([9 9],[1 3],[7 7]);
names={'observed','null','reversed'};
truths=repmat(struct('name','','A',zeros(9),'max_real_eigenvalue',NaN),3,1);
for i=1:3
    a=A; shift=(i-1)*h/2; a(ff)=a(ff)+shift; a(fb)=a(fb)-shift;
    eigmax=max(real(eig(a-.5*eye(9))));
    assert(eigmax<0,['Unstable truth: ' names{i} ' (no automatic rescaling)']);
    truths(i)=struct('name',names{i},'A',a,'max_real_eigenvalue',eigmax);
end
m=path_metrics(truths(2).A,1,3); assert(abs(m(3))<1e-12,'Null construction failed');
m=path_metrics(truths(3).A,1,3); assert(abs(m(3)+h)<1e-12,'Reversal construction failed');
end

function m=path_metrics(A,lo,hi)
s=3*(lo-1); t=3*(hi-1);
ff=mean([A(t+1,s+3),A(t+2,s+3)]);
fb=mean([A(s+1,t+1),A(s+3,t+1)]); m=[ff fb fb-ff];
end

function row=metrics(row,truth,estimate)
names={'EVC_PPC','EVC_cauPFC','PPC_cauPFC'}; pairs=[1 2;1 3;2 3];
for i=1:3
    a=path_metrics(truth,pairs(i,1),pairs(i,2)); b=path_metrics(estimate,pairs(i,1),pairs(i,2));
    for j=1:3
        labels={'FF','FB','H'}; key=[names{i} '_' labels{j}];
        row.(['true_' key])=a(j); row.(['estimated_' key])=b(j); row.(['bias_' key])=b(j)-a(j);
    end
end
mask=~eye(9); d=estimate(mask)-truth(mask);
row.offdiag_rmse=sqrt(mean(d.^2)); row.offdiag_bias=mean(d); row.offdiag_pearson_r=NaN;
row.offdiag_sign_accuracy=NaN; row.endpoint_H_sign_recovered=NaN;
if row.success
    r=corrcoef(truth(mask),estimate(mask)); row.offdiag_pearson_r=r(1,2);
    nz=mask & abs(truth)>1e-8; row.offdiag_sign_accuracy=mean(sign(estimate(nz))==sign(truth(nz)));
    h=row.true_EVC_cauPFC_H;
    if abs(h)>1e-10, row.endpoint_H_sign_recovered=double(sign(row.estimated_EVC_cauPFC_H)==sign(h)); end
end
end

function write_truth(out,truths,labels)
rows={};
for i=1:3
    for t=1:9
        for s=1:9
            rows{end+1,1}=struct('truth',truths(i).name,'target',t,'source',s, ...
                'target_label',char(labels{t}),'source_label',char(labels{s}), ...
                'A_parameter',truths(i).A(t,s),'A_effective',truths(i).A(t,s)-.5*(t==s), ...
                'max_real_eigenvalue',truths(i).max_real_eigenvalue); %#ok<AGROW>
        end
    end
end
writetable(records_table(vertcat(rows{:})),fullfile(out,'truth_matrices.csv'));
end

function tbl=records_table(records)
% Each struct is one record, including a scalar with empty text fields.
tbl=struct2table(records(:),'AsArray',true);
assert(height(tbl)==numel(records),'Record/table row count mismatch');
end

function selftest()
a=zeros(9); a(7,3)=.02; a(8,3)=.04; a(1,7)=.1; a(3,7)=.08;
t=make_truths(a); assert(max(abs(path_metrics(a,1,3)-[.03 .09 .06]))<1e-12);
assert(numel(t)==3 && isequal(size(t),[3 1]),'Truth struct allocation failed');
for i=1:3
    diff=t(i).A-a; mask=true(9); mask([sub2ind([9 9],[7 8],[3 3]) sub2ind([9 9],[1 3],[7 7])])=false;
    assert(all(diff(mask)==0),'Non-endpoint edges changed');
end
r=metrics(struct('success',true),a,a); assert(r.offdiag_rmse==0 && r.endpoint_H_sign_recovered==1);
r=metrics(struct('success',false),a,NaN(9)); assert(isnan(r.offdiag_rmse));
assert(~isequal(path_metrics(a',1,3),path_metrics(a,1,3)),'Direction check ineffective');
record=struct('case_id','one','last_warning','','error','','success',true,'H',.06);
one=records_table(record); assert(height(one)==1,'Scalar record export failed');
failed=record; failed.case_id='longer_case_id'; failed.success=false;
failed.error=sprintf('line one\nline two'); failed.H=NaN;
two=records_table([record;failed]); assert(height(two)==2,'Array record export failed');
file=[tempname '.csv']; cleanup=onCleanup(@() delete(file));
writetable(one,file);
loaded=readtable(file,'TextType','string'); assert(height(loaded)==1,'Scalar CSV roundtrip failed');
writetable(two,file);
loaded=readtable(file,'TextType','string'); assert(height(loaded)==2,'Multiline CSV roundtrip failed');
assert(loaded.H(1)==.06 && isnan(loaded.H(2)),'CSV numeric values changed');
end

function h=sha(file)
f=fopen(file,'rb'); assert(f>=0,['Cannot read ' file]); cleanup=onCleanup(@() fclose(f));
md=java.security.MessageDigest.getInstance('SHA-256');
while true
    bytes=fread(f,1024*1024,'*uint8'); if isempty(bytes), break; end
    md.update(typecast(bytes(:),'int8'));
end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
function h=sha_text(text)
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(typecast(unicode2native(text,'UTF-8'),'int8'));
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
function write_json(file,value)
f=fopen(file,'w'); assert(f>=0,['Cannot write ' file]); cleanup=onCleanup(@() fclose(f));
fprintf(f,'%s\n',jsonencode(value));
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
