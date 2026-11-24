function fit_roi_size(job_file)
% Serial fitting worker. Only writes to newly created attempt directories.
options_self_test();
if strcmp(job_file,'--self-test'), fprintf('PASS: returned-option comparison\n'); return; end
jobs=jsondecode(fileread(job_file));
for ji=1:numel(jobs)
 j=jobs(ji); dest=j.attempt_dir; result=struct('status','FAILED'); timer=tic;
 diary(fullfile(dest,'matlab_console.log')); lastwarn('');
 try
  assert(isfolder(j.spm_path) && isfolder(fullfile(j.tapas_path,'rDCM')), 'Configure valid toolbox paths');
  addpath(genpath(fullfile(j.tapas_path,'rDCM'))); addpath(j.spm_path);
  assert(startsWith(which('spm_detrend'),j.spm_path),'SPM path shadowed');
  assert(startsWith(which('tapas_rdcm_estimate'),j.tapas_path),'TAPAS path shadowed');
  spm('defaults','fMRI'); spm_jobman('initcfg'); rng(j.matlab_seed,'twister');
  t=load(j.input);ts=double(t.ts);names=cellstr(string(t.labels(:)))';
  n=size(ts,2);v=size(ts,1);assert(n==6 || n==9);assert(v==218);
  if n==6
   expected={'EVC_deep','EVC_mid','EVC_sup','cau_PFC_deep','cau_PFC_mid','cau_PFC_sup'};
  else
   expected={'EVC_deep','EVC_mid','EVC_sup','PPC_deep','PPC_mid','PPC_sup','cauPFC_deep','cauPFC_mid','cauPFC_sup'};
  end
  assert(isequal(names,expected),'Frozen node order mismatch');
  assert(all(isfinite(ts(:))) && all(std(ts,0,1)>=1e-8),'Invalid input time series');
  clear DCM options output
  DCM.n=n;DCM.v=v;DCM.Y.dt=double(t.TR);DCM.Y.X0=ones(v,1);
  DCM.Y.Q=spm_Ce(repmat(v,1,n));DCM.Y.y=ts;DCM.Y.name=names;
  DCM.U.u=zeros(v*16,1);DCM.U.name={'null'};DCM.U.dt=double(t.TR)/16;
  DCM.delays=repmat(double(t.TR)/2,n,1);DCM.TE=double(t.TE);
  DCM.options=struct('nonlinear',0,'two_state',0,'stochastic',0,'nograph',1,'wide_priors',0);
  DCM.a=ones(n);DCM.b=zeros(n,n,1);DCM.c=zeros(n,1);DCM.d=zeros(n,n,0);
  options=struct('scale',1,'estimateVL',0,'u_shift',0,'filter_str',4,'compute_signal',0,'wide_priors',0);
  requested_options=options;
  DCM.Y.y=spm_detrend(DCM.Y.y);scale_factor=4/max(max(DCM.Y.y(:))-min(DCM.Y.y(:)),4);
  DCM.Y.y=DCM.Y.y*scale_factor;
  [output,options]=tapas_rdcm_estimate(DCM,'r',options,1);
  A=output.Ep.A;assert(isequal(size(A),[n,n]) && all(isfinite(A(:))),'Invalid fitted A');
  result=struct('status','SUCCESS','A',A,'scale_factor',scale_factor,'seconds',toc(timer),'matrix_convention','A(target,source)','labels',{names});
  result.runtime=struct('matlab',version,'spm_file',which('spm_detrend'),'spm_sha256',filehash(which('spm_detrend')), ...
   'tapas_file',which('tapas_rdcm_estimate'),'tapas_sha256',filehash(which('tapas_rdcm_estimate')),'spm_Ce_sha256',filehash(which('spm_Ce')));
  if strcmp(j.phase,'baseline')
   old=load(j.original_output,'output','DCM','options');
   result.original_A=old.output.Ep.A;assert(isequal(size(result.original_A),[n,n]) && all(isfinite(result.original_A(:))),'Invalid original A');
   assert(isequal(size(DCM.Y.y),size(old.DCM.Y.y)),'Original normalized input shape mismatch');
   result.normalized_input_max_error=max(abs(DCM.Y.y(:)-old.DCM.Y.y(:)));
   result.normalized_input_pass=isequal(size(DCM.Y.y),size(old.DCM.Y.y)) && ...
    all(abs(DCM.Y.y(:)-old.DCM.Y.y(:))<=1e-8+1e-6*abs(old.DCM.Y.y(:)));
   result.options_pass=isequal(DCM.a,old.DCM.a) && isequal(DCM.b,old.DCM.b) && isequal(DCM.c,old.DCM.c) && ...
    isequal(DCM.d,old.DCM.d) && isequal(DCM.options,old.DCM.options) && ...
    isequal(DCM.Y.dt,old.DCM.Y.dt) && isequal(DCM.TE,old.DCM.TE) && isequal(DCM.delays,old.DCM.delays) && ...
    isequal(DCM.Y.X0,old.DCM.Y.X0) && isequal(DCM.Y.Q,old.DCM.Y.Q) && isequal(DCM.U,old.DCM.U);
   [returned_ok,result.returned_option_checks]=compare_returned_options(requested_options,options,old.options);
   result.options_pass=result.options_pass && returned_ok;
   result.requested_options=requested_options;
   result.original_requested_options_evidence='Original MAT does not store requested_options. Input values are supported by reviewed original source; missing returned fields are not retrospective runtime verification.';
   result.original_labels_match=isequal(cellstr(string(old.DCM.Y.name(:)))',names);
   result.options_pass=result.options_pass && result.original_labels_match;
   result.original_output_sha256=filehash(j.original_output);
  end
  save(fullfile(dest,'rDCM_output.mat'),'output','options','DCM','requested_options','scale_factor','-v7.3');
 catch err
  result=struct('status','FAILED','error',getReport(err,'extended','hyperlinks','off'),'seconds',toc(timer));
 end
 [w,wi]=lastwarn;result.warning=w;result.warning_id=wi;result.config_token=j.config_token;result.input_sha256=j.input_sha256;
 f=fopen(fullfile(dest,'result.json.tmp'),'w');assert(f>=0);fwrite(f,jsonencode(result),'char');fclose(f);
 movefile(fullfile(dest,'result.json.tmp'),fullfile(dest,'result.json'));diary off;
end
end
function [ok,checks]=compare_returned_options(requested,current,original)
expected=struct('scale',1,'estimateVL',0,'u_shift',0,'filter_str',4,'compute_signal',0,'wide_priors',0);
assert(isequal(requested,expected),'Requested model options changed');
% Server diagnostics found these three fields absent in BOTH returned structs.
% Do not infer their historical input values from that absence.
not_retained={'scale','estimateVL','wide_priors'};
ok=true; checks=struct(); fields=fieldnames(expected);
for fi=1:numel(fields)
 f=fields{fi}; has_new=isfield(current,f); has_old=isfield(original,f);
 if has_new && has_old
  equal=isequal(current.(f),original.(f)); verdict='EQUAL';
  if ~equal, verdict='VALUE_MISMATCH'; end
 elseif ~has_new && ~has_old && any(strcmp(f,not_retained))
  equal=true; verdict='NOT_RETAINED_BOTH_SOURCE_EVIDENCE_ONLY';
 else
  equal=false; verdict='MISSING_OR_ASYMMETRIC';
 end
 checks.(f)=struct('current_present',has_new,'original_present',has_old,'status',verdict);
 ok=ok && equal;
end
end
function options_self_test()
q=struct('scale',1,'estimateVL',0,'u_shift',0,'filter_str',4,'compute_signal',0,'wide_priors',0);
r=rmfield(q,{'scale','estimateVL','wide_priors'});
[ok,c]=compare_returned_options(q,r,r); assert(ok && strcmp(c.scale.status,'NOT_RETAINED_BOTH_SOURCE_EVIDENCE_ONLY'));
assert(compare_returned_options(q,q,q));
bad=r; bad.filter_str=3; assert(~compare_returned_options(q,r,bad));
bad=r; bad.scale=1; assert(~compare_returned_options(q,r,bad));
bad=rmfield(r,'filter_str'); assert(~compare_returned_options(q,bad,bad));
end
function h=filehash(p)
f=fopen(p,'rb');assert(f>=0);c=onCleanup(@()fclose(f));md=java.security.MessageDigest.getInstance('SHA-256');
while ~feof(f),b=fread(f,1048576,'*uint8');md.update(b);end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
