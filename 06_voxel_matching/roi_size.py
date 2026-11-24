#!/usr/bin/env python3
"""Voxel-count sensitivity: explicit inputs, immutable run, serial bounded jobs."""
import argparse, csv, hashlib, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat

STAGES={'EVC':['V1','V2','V3'],'PPC':['LIPv','LIPd','VIP','MIP','IP0','IP1','IP2','7PC','7Am','7AL','7PL','7Pm','7m'],'cauPFC':['6d','6v','6mp','FEF','PEF','55b','SFL']}
LAYERS=['deep','mid','sup']; MODELS={'six':['EVC','cauPFC'],'nine':['EVC','PPC','cauPFC']}
SUBJECTS=[f'sub-{i:02d}' for i in list(range(1,15))+[16]]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def plain(x):
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(plain(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def csvread(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status','reason']
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([plain(r) for r in rows])
def expand(c,value,s=''):
 if not value:raise ValueError('Required path not configured')
 return Path(value.format(project_root=c['project_root'],subject=s)).expanduser().resolve()
def labels(model):return [f'{"cau_PFC" if model=="six" and g=="cauPFC" else g}_{d}' for g in MODELS[model] for d in LAYERS]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def token(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()+Path(__file__).read_bytes()+Path(__file__).with_name('fit_roi_size.m').read_bytes()+Path(__file__).with_name('summarize_roi_size.py').read_bytes()).hexdigest()
def fingerprint(p,full=False):
 p=Path(p);st=p.stat();return {'path':str(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha(p) if full or st.st_size<8*1024*1024 else None}
def compare(x,y,c,A=False):
 x=np.asarray(x,float);y=np.asarray(y,float)
 require(x.shape==y.shape,f'parity shape {x.shape} != {y.shape}')
 require(np.isfinite(x).all() and np.isfinite(y).all(),'nonfinite parity input')
 err=np.abs(x-y);std=np.std(y,axis=0) if y.ndim==2 else np.std(y)
 norm=float(np.max(np.sqrt(np.mean((x-y)**2,axis=0))/np.maximum(std,1e-8)))
 t=c['tolerance'];ok=np.allclose(x,y,atol=t['A_atol' if A else 'ts_atol'],rtol=t['A_rtol' if A else 'ts_rtol'])
 if not A:ok=ok and norm<=t['ts_rmse_over_sd']
 return dict(pass_check=bool(ok),max_abs_error=float(err.max()),rmse=float(np.sqrt(np.mean(err**2))),max_node_rmse_over_reference_sd=norm)

def allocations(capacities,min_parcel=5):
 """Maximum common B with parcel quotas differing by <=1 within each node."""
 limits={k:len(v)*min(v)+sum(n>min(v) for n in v) for k,v in capacities.items()}
 B=min(limits.values());quotas={}
 require(B>=max(len(v)*min_parcel for v in capacities.values()),f'Low feasible budget B={B}; required common minimum={max(len(v)*min_parcel for v in capacities.values())}; limiting nodes={[str(k) for k,v in limits.items() if v==B]}; capacities={capacities}. Check masks/coverage; do not delete parcels or subjects.')
 for k,v in capacities.items():
  q,r=divmod(B,len(v));a=[q]*len(v)
  eligible=[i for i,n in enumerate(v) if n>=q+1]
  require(q>=min_parcel and len(eligible)>=r,f'low budget {B}; {k}: floor quota={q}, minimum={min_parcel}; no parcel may be removed')
  for i in eligible[:r]:a[i]+=1
  require(sum(a)==B and all(a[i]<=v[i] for i in range(len(v))),'quota capacity error');quotas[k]=a
 return B,quotas,limits

def hemi_quota(q,L,R):
 require(L+R>=q and q>0,'bad hemisphere capacity')
 ideal=q*L/(L+R);lo=max(1 if L else 0,q-R);hi=min(L,q-(1 if R else 0))
 require(lo<=hi,'cannot preserve every originally present hemisphere')
 l=min(hi,max(lo,int(math.floor(ideal+.5))))
 return l,q-l,ideal

def derived_seed(c,s,model,rep):
 return int.from_bytes(hashlib.sha256(f'{c["master_seed"]}|{s}|{model}|{rep}'.encode()).digest()[:8],'little')

def sources(c,s):
 base=expand(c,c['step3'],s); meta_path=base/'pial_exclusion_metadata.json'
 meta=readj(meta_path) if meta_path.is_file() else {}
 bold=expand(c,c['bold_override'],s) if c.get('bold_override') else (Path(meta['bold_file']).resolve() if meta.get('bold_file') else None)
 hemi=expand(c,c['hemisphere_images'].get(s),s) if c['hemisphere_images'].get(s) else None
 return dict(base=base,metadata=meta_path,bold=bold,hemi=hemi,roi_names=base/'roi_names.npy',ts=base/'ts_matrix.npy',counts=base/'n_voxels_clean.npy',**{d:base/'clean_masks'/f'layer{i+1}_clean.nii.gz' for i,d in enumerate(LAYERS)})

def inspect(c,out):
 rows=[]
 for s in SUBJECTS:
  try:
   for key,p in sources(c,s).items():
    if key!='base':rows.append(dict(subject=s,input=key,path=str(p) if p else '',status='FOUND' if p and p.is_file() else 'MISSING_OR_UNCONFIGURED'))
  except Exception as e:rows.append(dict(subject=s,input='source_metadata',status='ERROR',reason=str(e)))
 for model in MODELS:
  for s in c['pilot_subjects']:
   for k in ['original_input','original_output']:
    p=expand(c,c[k][model],s);rows.append(dict(subject=s,input=model+'_'+k,path=str(p),status='FOUND' if p.is_file() else 'MISSING'))
 for k in ['spm_path','tapas_path']:
  p=Path(c[k]);rows.append(dict(subject='all',input=k,path=str(p),status='FOUND' if p.is_dir() else 'MISSING'))
 for k in ['six_metrics','behavior_table','nine_edges']:
  p=expand(c,c[k]);rows.append(dict(subject='all',input=k,path=str(p),status='FOUND' if p.is_file() else 'MISSING'))
 csvwrite(out/'input_inventory.csv',rows)
 writej(out/'inspect_status.json',{'status':'READY_TO_CONFIGURE_OR_PREPARE','missing':sum(r['status']!='FOUND' for r in rows),'note':'No MRI read. Hemisphere inputs must be supplied from reliable anatomical labels in the extraction grid, not inferred from x=0.'})
 print('Read',out/'input_inventory.csv')

def read_subject(c,s,folder):
 import nibabel as nib
 src=sources(c,s)
 for k,p in src.items():
  if k!='base':require(p is not None and p.is_file(),f'{s}: missing/unconfigured {k}: {p}')
 names=np.load(src['roi_names'],allow_pickle=c['trusted_roi_name_pickle']).astype(str)
 names=np.array([c['roi_aliases'].get(x,x) for x in names]);require(len(names)==len(set(names)),'duplicate ROI labels')
 selected=[g for group in STAGES.values() for g in group];require(len(set(selected))==23,'overlapping stage membership')
 require(set(selected)<=set(names),'missing frozen parcels')
 reference=np.load(src['ts'],allow_pickle=False);oldcounts=np.load(src['counts'],allow_pickle=False)
 require(reference.shape==(len(names),3,218),'unexpected stored ROI/depth/time shape')
 img=nib.load(src['bold'],keep_file_open=True);require(img.shape[-1]==218 and len(img.shape)==4,'unexpected BOLD dimensions')
 def volume(p):
  v=nib.load(p);require(v.shape==img.shape[:3] and np.allclose(v.affine,img.affine,atol=1e-4,rtol=0),f'grid/affine mismatch: {p}')
  a=np.asarray(v.dataobj);require(np.isfinite(a).all() and np.allclose(a,np.rint(a),atol=1e-5,rtol=0),f'invalid labels: {p}')
  return np.rint(a).astype(np.int32).ravel(order='C')
 h=volume(src['hemi']);lh=c['hemisphere_values']['L'];rh=c['hemisphere_values']['R'];require(lh!=rh and lh and rh,'distinct nonzero hemisphere codes required')
 masks=[volume(src[d]) for d in LAYERS]
 require(all(m.min()>=0 and m.max()<=len(names) for m in masks),'mask ROI label range')
 require(np.max(np.sum([m>0 for m in masks],axis=0))<=1,'overlapping depth masks')
 segments=[];index=[];start=0;rows=[]
 for g,members in STAGES.items():
  for d,mask in zip(LAYERS,masks):
   for roi in members:
    ri=list(names).index(roi);flat=np.flatnonzero(mask==ri+1)
    require(len(flat)==int(oldcounts[ri,LAYERS.index(d)]),f'count mismatch {s}/{roi}/{d}')
    require(len(flat)>=5,f'insufficient original count {s}/{roi}/{d}')
    require(np.isin(h[flat],[lh,rh]).all(),f'unassigned/ambiguous hemisphere {s}/{roi}/{d}')
    seg=dict(stage=g,parcel=roi,depth=d,roi_index=ri,start=start,stop=start+len(flat))
    segments.append(seg);index.extend(flat);start+=len(flat)
    for hemi,val in [('L',lh),('R',rh)]:rows.append(dict(subject=s,stage=g,parcel=roi,depth=d,hemisphere=hemi,n_mask=int(np.sum(h[flat]==val))))
 index=np.array(index,np.int64);require(len(set(index))==len(index),'duplicate extraction indices')
 folder.mkdir(parents=True,exist_ok=True)
 # Read one 3D volume at a time. Repeated compressed reads can be slow, but peak memory is bounded.
 vox=np.lib.format.open_memmap(folder/'voxels.npy',mode='w+',dtype=np.float32,shape=(len(index),218))
 for t in range(218):
  vox[:,t]=np.asarray(img.dataobj[...,t],dtype=np.float32).ravel(order='C')[index]
  if t%50==0:print(s,'BOLD volume',t+1,'/218',flush=True)
 vox.flush();finite=np.isfinite(vox).all(axis=1)
 np.save(folder/'flat_indices.npy',index);np.save(folder/'hemisphere.npy',h[index]);np.save(folder/'affine.npy',img.affine)
 reconstructed=np.empty((23,3,218),np.float32);lookup={r:i for i,r in enumerate(selected)};parity=[];signal_qc=[]
 for seg in segments:
  a=vox[seg['start']:seg['stop']];good=finite[seg['start']:seg['stop']]
  for row in rows:
   if (row['parcel'],row['depth'])==(seg['parcel'],seg['depth']):
    maskh=h[index[seg['start']:seg['stop']]]==c['hemisphere_values'][row['hemisphere']]
    row['n_valid']=int(np.sum(good & maskh));row['n_invalid']=row['n_mask']-row['n_valid']
  ts=a.mean(axis=0,dtype=np.float32);sd=float(ts.std(ddof=1));signal_qc.append(dict(subject=s,stage=seg['stage'],parcel=seg['parcel'],depth=seg['depth'],n_voxels=len(a),n_constant_voxels=int(np.sum(np.std(a,axis=1)==0)),mean=float(ts.mean()),sd=sd,tsnr=float(ts.mean()/sd) if c['tsnr_baseline_confirmed'] and sd>0 else None,tsnr_stage=c['signal_stage'] if c['tsnr_baseline_confirmed'] else 'NOT_ESTABLISHED'));reconstructed[lookup[seg['parcel']],LAYERS.index(seg['depth'])]=ts
  if good.all():res=compare(ts,reference[seg['roi_index'],LAYERS.index(seg['depth'])],c)
  else:res={'pass_check':False,'reason':'Nonfinite raw voxels; not silently dropped or imputed'}
  parity.append(dict(subject=s,parcel=seg['parcel'],depth=seg['depth'],**res))
 csvwrite(folder/'counts.csv',rows);csvwrite(folder/'parcel_parity.csv',parity);csvwrite(folder/'original_signal_qc.csv',signal_qc)
 require(finite.all(),'nonfinite voxel signals; counts reported, repair lineage before sampling')
 require(all(r['pass_check'] for r in parity),'all-voxel parcel baseline mismatch; see parcel_parity.csv')
 np.save(folder/'parcel_ts.npy',reconstructed)
 baseline_qc=[]
 for model in MODELS:
  ts=nodes(reconstructed,selected,model)
  orig=loadmat(expand(c,c['original_input'][model],s),squeeze_me=True)
  require(orig['ts'].shape==ts.shape,'original model input shape')
  ol=[str(v) for v in np.asarray(orig['labels']).ravel()]
  require(ol==labels(model),'original node label order mismatch')
  require(close_scalar(orig['TR'],2.2) and close_scalar(orig['TE'],.026),'TR/TE mismatch')
  row=dict(subject=s,model=model,**compare(ts,orig['ts'],c));baseline_qc.append(row)
  csvwrite(folder/'node_parity.csv',baseline_qc)
  require(row['pass_check'],f'{model}: all-voxel model-input baseline mismatch')
  savemat(folder/f'{model}_baseline.mat',dict(ts=ts,labels=np.array(labels(model),object),TR=2.2,TE=.026))
 csvwrite(folder/'node_parity.csv',baseline_qc)
 nodeqc=[]
 for model in MODELS:
  t=loadmat(folder/f'{model}_baseline.mat')['ts']
  for i,lab in enumerate(labels(model)):
   a=t[:,i];nodeqc.append(dict(subject=s,model=model,node=lab,mean=float(a.mean()),sd=float(a.std(ddof=1)),tsnr=float(a.mean()/a.std(ddof=1)) if c['tsnr_baseline_confirmed'] else None,stage=c['signal_stage'] if c['tsnr_baseline_confirmed'] else 'NOT_ESTABLISHED'))
 csvwrite(folder/'original_node_signal_qc.csv',nodeqc)
 info=dict(subject=s,segments=segments,parcels=selected,shape=img.shape,affine=img.affine,source_dtype=str(img.get_data_dtype()) if hasattr(img,'get_data_dtype') else 'mock-test',hemisphere_provenance=c['hemisphere_provenance'],flat_index_order='zero-based C order; voxel ijk = unravel_index(index, shape[:3], order=C)',source_fingerprints=[fingerprint(p,c['hash_large_inputs']) for k,p in src.items() if k!='base'],voxel_sha256=sha(folder/'voxels.npy'))
 info['cache_hashes']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
 writej(folder/'source.json',info);return rows,parity,baseline_qc

def close_scalar(a,b):return bool(np.isclose(float(a),b,rtol=0,atol=1e-12))
def nodes(parcels,names,model):
 out=[]
 for g in MODELS[model]:
  a=parcels[[names.index(r) for r in STAGES[g]]]
  for i in range(3):
   require(np.isfinite(a[:,i]).all() and np.all(np.std(a[:,i],axis=1)>=1e-6),'invalid parcel signal')
   out.append(a[:,i].mean(axis=0,dtype=np.float32))
 ts=np.stack(out,axis=1).astype(np.float64);require(np.all(np.std(ts,axis=0)>=1e-6),'invalid composite');return ts

def plan_subject(c,s,counts):
 lookup={(r['parcel'],r['depth'],r['hemisphere']):r['n_valid'] for r in counts};rows=[];summary=[]
 for model,stages in MODELS.items():
  cap={(g,d):[lookup[r,d,'L']+lookup[r,d,'R'] for r in STAGES[g]] for g in stages for d in LAYERS}
  B,quotas,limits=allocations(cap,c['minimum_parcel_quota'])
  for (g,d),qq in quotas.items():
   for roi,q in zip(STAGES[g],qq):
    L,R=lookup[roi,d,'L'],lookup[roi,d,'R'];l,r,ideal=hemi_quota(q,L,R)
    for hemi,n,take,ideal_h in [('L',L,l,ideal),('R',R,r,q-ideal)]:rows.append(dict(subject=s,model=model,stage=g,depth=d,parcel=roi,hemisphere=hemi,node_budget=B,parcel_quota=q,original_count=n,quota=take,retained_fraction=take/n if n else None,ideal_proportional_quota=ideal_h,rounding_delta=take-ideal_h,original_left_fraction=L/(L+R),sampled_left_fraction=l/q))
  summary.append(dict(subject=s,model=model,node_budget=B,minimum_parcel_quota=min(min(v) for v in quotas.values()),limiting_nodes=';'.join(f'{g}:{d}' for (g,d),v in limits.items() if v==B),limiting_parcels=';'.join(f'{g}:{d}:{STAGES[g][i]}' for (g,d),v in limits.items() if v==B for i,n in enumerate(cap[g,d]) if n==min(cap[g,d])),rule='maximum_common_node_budget_balanced_parcels',coverage_note='AT_MINIMUM_EXTRACTION_FLOOR_REVIEW' if min(min(v) for v in quotas.values())==c['minimum_parcel_quota'] else 'review_counts_and_retention_before_formal'))
 return rows,summary

def prepare(c,out):
 writej(out/'prepare_status.json',{'status':'RUNNING','config_token':token(c)})
 allcounts=[];allparity=[];nodeparity=[];plans=[];budgets=[];status=[]
 for s in SUBJECTS:
  folder=out/'cache'/s
  try:
   if (folder/'source.json').exists():
    info=readj(folder/'source.json');require(sha(folder/'voxels.npy')==info['voxel_sha256'],'cached voxels changed')
    for fp in info['source_fingerprints']:require(fingerprint(fp['path'],c['hash_large_inputs'])==fp,'source changed')
    cc=csvread(folder/'counts.csv');pp=csvread(folder/'parcel_parity.csv');nn=csvread(folder/'node_parity.csv')
    for r in cc:
     for k in ['n_mask','n_valid','n_invalid']:r[k]=int(r[k])
   else:
    require(not folder.exists(),f'partial cache {folder}; preserve diagnostics and use a new run directory')
    cc,pp,nn=read_subject(c,s,folder)
   allcounts+=cc;allparity+=pp;nodeparity+=nn
   a,b=plan_subject(c,s,cc);plans+=a;budgets+=b;status.append(dict(subject=s,status='PASS',reason=''))
  except Exception as e:status.append(dict(subject=s,status='BLOCKED',reason=str(e)))
  csvwrite(out/'feasibility.csv',status);csvwrite(out/'counts.csv',allcounts);csvwrite(out/'parcel_parity.csv',allparity);csvwrite(out/'node_parity.csv',nodeparity);csvwrite(out/'sampling_plan.csv',plans);csvwrite(out/'budgets.csv',budgets)
 node_counts=[]
 for s in SUBJECTS:
  for m,stages in MODELS.items():
   for g in stages:
    for d in LAYERS:
     a=[r for r in plans if r['subject']==s and r['model']==m and r['stage']==g and r['depth']==d]
     if a:node_counts.append(dict(subject=s,model=m,stage=g,depth=d,original_total=sum(r['original_count'] for r in a),matched_total=sum(r['quota'] for r in a),original_L=sum(r['original_count'] for r in a if r['hemisphere']=='L'),matched_L=sum(r['quota'] for r in a if r['hemisphere']=='L'),original_R=sum(r['original_count'] for r in a if r['hemisphere']=='R'),matched_R=sum(r['quota'] for r in a if r['hemisphere']=='R'),original_signal_left_weight=float(np.mean([r['original_left_fraction'] for r in a if r['hemisphere']=='L'])),matched_signal_left_weight=float(np.mean([r['sampled_left_fraction'] for r in a if r['hemisphere']=='L']))))
 csvwrite(out/'node_counts.csv',node_counts)
 ready=len(status)==15 and all(r['status']=='PASS' for r in status)
 if ready:writej(out/'cache_manifests.json',{s:sha(out/'cache'/s/'source.json') for s in SUBJECTS})
 writej(out/'prepare_status.json',dict(status='PASS' if ready else 'BLOCKED',config_token=token(c),subjects=status))
 require(ready,'Feasibility blocked; inspect per-subject cache QC and feasibility.csv. No subjects silently excluded.')


def make_jobs(c,out,phase):
 subjects=c['pilot_subjects'] if phase in ['baseline','pilot'] else SUBJECTS
 n=1 if phase=='baseline' else (3 if phase=='pilot' else c['formal_repeats']);jobs=[];plan=csvread(out/'sampling_plan.csv')
 for s in subjects:
  cache=out/'cache'/s;info=readj(cache/'source.json')
  require(sha(cache/'voxels.npy')==info['voxel_sha256'],'cache changed')
  for name,digest in info['cache_hashes'].items():require(sha(cache/name)==digest,'cached extraction file changed: '+name)
  for fp in info['source_fingerprints']:require(fingerprint(fp['path'],c['hash_large_inputs'])==fp,'source changed; new run required')
  vox=np.load(cache/'voxels.npy',mmap_mode='r');hemi=np.load(cache/'hemisphere.npy');indices=np.load(cache/'flat_indices.npy')
  for model in MODELS:
   for rep in range(n):
    job=out/phase/s/model/f'rep-{rep:03d}';job.mkdir(parents=True,exist_ok=True)
    spec=job/'job.json'
    if spec.exists():
     rec=readj(spec);require(rec['config_token']==token(c) and sha(job/'input.mat')==rec['input_sha256'],'existing job input changed');
     if phase!='baseline':require(sha(job/'sample_indices.npz')==rec['sample_indices_sha256'],'stored sample indices changed')
     jobs.append(rec);continue
    if phase=='baseline':
     import shutil
     shutil.copyfile(cache/f'{model}_baseline.mat',job/'input.mat');seed=derived_seed(c,s,model,-1)
    else:
     seed=derived_seed(c,s,model,rep);rng=np.random.default_rng(seed);parcel=np.load(cache/'parcel_ts.npy').copy();saved={};qc=[]
     for seg in info['segments']:
      if seg['stage'] not in MODELS[model]:continue
      used=[]
      for side in ['L','R']:
       row=next(r for r in plan if r['subject']==s and r['model']==model and r['parcel']==seg['parcel'] and r['depth']==seg['depth'] and r['hemisphere']==side)
       candidates=np.arange(seg['start'],seg['stop']);candidates=candidates[hemi[candidates]==c['hemisphere_values'][side]]
       take=rng.choice(candidates,size=int(row['quota']),replace=False);used.extend(take.tolist());saved[f'{seg["parcel"]}__{seg["depth"]}__{side}']=indices[take]
      used=np.array(sorted(used));a=vox[used].mean(axis=0,dtype=np.float32);parcel[info['parcels'].index(seg['parcel']),LAYERS.index(seg['depth'])]=a
      qc.append(dict(subject=s,model=model,rep=rep,parcel=seg['parcel'],depth=seg['depth'],n_original=seg['stop']-seg['start'],n_sampled=len(used),mean=float(a.mean()),sd=float(a.std(ddof=1)),tsnr=float(a.mean()/a.std(ddof=1)) if c['tsnr_baseline_confirmed'] else None,tsnr_stage=c['signal_stage'] if c['tsnr_baseline_confirmed'] else 'NOT_ESTABLISHED'))
     np.savez_compressed(job/'sample_indices.npz',**saved);csvwrite(job/'signal_qc.csv',qc)
     ts=nodes(parcel,info['parcels'],model)
     nodeqc=[]
     for i,lab in enumerate(labels(model)):
      a=ts[:,i];nodeqc.append(dict(subject=s,model=model,rep=rep,node=lab,mean=float(a.mean()),sd=float(a.std(ddof=1)),tsnr=float(a.mean()/a.std(ddof=1)) if c['tsnr_baseline_confirmed'] else None,stage=c['signal_stage'] if c['tsnr_baseline_confirmed'] else 'NOT_ESTABLISHED'))
     csvwrite(job/'node_signal_qc.csv',nodeqc);savemat(job/'input.mat',dict(ts=ts,labels=np.array(labels(model),object),TR=2.2,TE=.026))
    rec=dict(subject=s,model=model,rep=rep,phase=phase,seed=str(seed),matlab_seed=int(seed%2**32),input=str(job/'input.mat'),input_sha256=sha(job/'input.mat'),job_dir=str(job),original_output=str(expand(c,c['original_output'][model],s)),spm_path=c['spm_path'],tapas_path=c['tapas_path'],config_token=token(c))
    if phase!='baseline':rec['sample_indices_sha256']=sha(job/'sample_indices.npz')
    writej(spec,rec);jobs.append(rec)
 return jobs


def execute(c,out,phase,retry=False):
 started=time.monotonic();export_started=started
 require(readj(out/'prepare_status.json')['status']=='PASS','prepare must pass for all 15')
 for s,digest in readj(out/'cache_manifests.json').items():require(sha(out/'cache'/s/'source.json')==digest,'cache manifest changed')
 if phase!='baseline':require(readj(out/'baseline_report.json')['status']=='PASS','baseline must pass')
 writej(out/f'{phase}_report.json',{'status':'RUNNING','config_token':token(c)})
 if phase=='formal':writej(out/'statistics_status.json',{'status':'PENDING_ALL_FITS','note':'Prior tables, if any, are not current until this status becomes COMPLETE'})
 jobs=make_jobs(c,out,phase);export_seconds=time.monotonic()-export_started;todo=[]
 for r in jobs:
  folder=Path(r['job_dir']);attempts=sorted(folder.glob('attempt-*'));latest=attempts[-1]/'result.json' if attempts else None
  if latest and latest.exists() and readj(latest).get('status')=='SUCCESS':continue
  if attempts and not retry:continue
  attempt=folder/f'attempt-{len(attempts)+1:03d}';attempt.mkdir();r=dict(r,attempt_dir=str(attempt));todo.append(r)
 joblist=out/f'{phase}_jobs_pending.json';writej(joblist,todo)
 if todo:
  worker=Path(__file__).parent.resolve();esc=lambda x:str(x).replace("'","''")
  cmd=[c['matlab_executable'],'-batch',f"addpath('{esc(worker)}'); fit_roi_size('{esc(joblist)}')"]
  with (out/f'{phase}_matlab_{time.time_ns()}.log').open('w') as log:
   try:
    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT);print('MATLAB return code',result.returncode)
   except OSError as e:
    log.write(str(e));print('MATLAB launch failed:',e)
 rows=[];baseline=[]
 for r in jobs:
  ats=sorted(Path(r['job_dir']).glob('attempt-*'));f=ats[-1]/'result.json' if ats else None
  data=readj(f) if f and f.exists() else dict(status='INTERRUPTED_OR_NOT_RUN')
  if data['status']=='SUCCESS':require(data['config_token']==token(c) and data['input_sha256']==r['input_sha256'],'result/input version mismatch')
  row=dict(subject=r['subject'],model=r['model'],rep=r['rep'],phase=phase,seed=r['seed'],attempt=str(ats[-1]) if ats else '',status=data['status'],warnings=data.get('warning',''),reason=data.get('error',''),seconds=data.get('seconds'))
  if data['status']=='SUCCESS':
   A=np.array(data['A']);require(A.shape==((6,6) if r['model']=='six' else (9,9)),'A shape')
   k=3 if r['model']=='six' else 6;FF=float((A[k,2]+A[k+1,2])/2);FB=float((A[0,k]+A[2,k])/2)
   row.update(FF=FF,FB=FB,H=FB-FF,scale_factor=data['scale_factor'])
   if phase=='baseline':
    oldA=np.array(data['original_A']);originals=csvread(expand(c,c['six_metrics'] if r['model']=='six' else c['behavior_table']))
    target=[v for v in originals if v['subject']==r['subject'] and (r['model']=='nine' or v['network']=='cauPFC_EVC' and v['condition']=='pial5_current')]
    require(len(target)==1,'baseline reference feature key')
    feature_ok=np.allclose([(oldA[k,2]+oldA[k+1,2])/2,(oldA[0,k]+oldA[2,k])/2],[float(target[0]['signed_FF']),float(target[0]['signed_FB'])],rtol=1e-10,atol=1e-12)
    baseline.append(dict(subject=r['subject'],model=r['model'],**compare(A,oldA,c,A=True),normalized_input_pass=data['normalized_input_pass'],normalized_input_max_error=data['normalized_input_max_error'],options_pass=data['options_pass'],original_features_match=bool(feature_ok)))
  rows.append(row)
 csvwrite(out/f'{phase}_fits.csv',rows)
 complete=all(r['status']=='SUCCESS' for r in rows)
 profiles=[]
 for r in rows:
  f=Path(r['attempt'])/'result.json'
  if r['status']=='SUCCESS':profiles.append(readj(f)['runtime'])
 runtime_ok=bool(profiles) and all(p==profiles[0] for p in profiles)
 runtime_file=out/'baseline_runtime.json'
 if phase=='baseline' and runtime_ok:writej(runtime_file,profiles[0])
 if phase!='baseline':runtime_ok=runtime_ok and runtime_file.is_file() and profiles[0]==readj(runtime_file)
 complete=complete and runtime_ok
 report=dict(status='PASS' if complete else 'BLOCKED',expected_jobs=len(jobs),successful=sum(r['status']=='SUCCESS' for r in rows),config_token=token(c),total_fit_seconds=sum(r['seconds'] or 0 for r in rows),warnings=sum(bool(r['warnings']) for r in rows),runtime_consistent=runtime_ok,job_preparation_seconds=export_seconds,phase_wall_seconds=time.monotonic()-started)
 if phase=='baseline':
  csvwrite(out/'baseline_A_parity.csv',baseline);report['status']='PASS' if complete and len(baseline)==4 and all(r['pass_check'] and r['normalized_input_pass'] and r['options_pass'] and r['original_features_match'] for r in baseline) else 'BLOCKED'
 if phase=='pilot':
  report['formal_jobs']=15*2*c['formal_repeats'];report['estimated_formal_fit_seconds']=report['total_fit_seconds']/len(jobs)*report['formal_jobs'] if complete else None
  size=sum(p.stat().st_size for r in jobs for p in Path(r['job_dir']).rglob('*') if p.is_file())
  report['pilot_job_bytes']=size;report['estimated_formal_job_bytes']=size/len(jobs)*report['formal_jobs'];report['estimate_note']='extrapolation from two preselected subjects; excludes cache, future retries, concurrent load and input extraction'
  report['pilot_report_review']='Review numerical parity, quotas, coverage, warnings, runtime, memory/disk; never choose budget/seed based on effect size or significance'
 writej(out/f'{phase}_report.json',report)
 quality=[]
 for r in jobs:
  p=Path(r['job_dir'])/'node_signal_qc.csv'
  if p.exists():quality+=csvread(p)
 csvwrite(out/f'{phase}_node_signal_qc.csv',quality)
 if phase=='formal' and complete:
  from summarize_roi_size import summarize
  try:summarize(c,out)
  except Exception as e:
   writej(out/'statistics_status.json',{'status':'BLOCKED','reason':str(e),'traceback':traceback.format_exc()});raise
 print(report)

def reference_checks(c,out):
 b=csvread(expand(c,c['behavior_table']));e=csvread(expand(c,c['nine_edges']));six=csvread(expand(c,c['six_metrics']))
 require(len(b)==15 and {r['subject'] for r in b}==set(SUBJECTS),'behavior subject IDs')
 ed={(r['subject'],int(r['target_index']),int(r['source_index'])):float(r['A']) for r in e}
 require(len(ed)==len(e)==15*81,'nine-node edge table cardinality/duplicates')
 require(set(ed)=={(s,t,u) for s in SUBJECTS for t in range(1,10) for u in range(1,10)},'nine-node edge keys')
 require(all(r['matrix_convention']=='A(target,source)' for r in e),'matrix convention')
 rows=[]
 for r in b:
  s=r['subject'];ff=(ed[s,7,3]+ed[s,8,3])/2;fb=(ed[s,1,7]+ed[s,3,7])/2
  ok=np.allclose([ff,fb],[float(r['signed_FF']),float(r['signed_FB'])],atol=1e-12,rtol=1e-10)
  rows.append(dict(subject=s,check='original_nine_edges_vs_behavior_features',pass_check=bool(ok)))
 for network in ['cauPFC_EVC','cauPFC_V1']:
  a=[r for r in six if r['network']==network and r['condition']=='pial5_current']
  require(len(a)==15 and {r['subject'] for r in a}==set(SUBJECTS),'original six-node metric cardinality')
 csvwrite(out/'reference_checks.csv',rows);require(all(r['pass_check'] for r in rows),'behavior input does not match nine-node features')


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['inspect','prepare','baseline','pilot','formal']);p.add_argument('--config',default='config.json');p.add_argument('--retry-failed',action='store_true');p.add_argument('--pilot-review-sha');a=p.parse_args();c=readj(a.config)
 require(c['subjects']==SUBJECTS and c['pilot_subjects']==['sub-01','sub-02'],'frozen participants; pilot chosen first two IDs before sensitivity results')
 out=expand(c,c['output']);require('roi_size_sensitivity' in out.parts,'output must be under new roi_size_sensitivity directory');require(not out.is_symlink(),'no symlink output');
 for s in SUBJECTS:
  b=expand(c,c['step3'],s);require(out!=b and b not in out.parents and out not in b.parents,'output overlaps original Step3')
 out.mkdir(parents=True,exist_ok=True)
 import fcntl
 process_lock=(out/'.process.lock').open('w');fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.mode=='inspect':inspect(c,out);return
 require(all(c['hemisphere_images'].get(s) for s in SUBJECTS),'Configure all hemisphere_images first; run inspect')
 require(bool(c.get('hemisphere_provenance')),'Document hemisphere label source and alignment in hemisphere_provenance')
 require(not c['tsnr_baseline_confirmed'] or c['signal_stage'] and 'UNCONFIRMED' not in c['signal_stage'],'Document actual mean-preserving BOLD stage before enabling tSNR')
 require(c['formal_repeats']>=1 and c['minimum_parcel_quota']>=5,'invalid repeats/quota')
 lock=out/'run_lock.json';state={'config_token':token(c),'config':c}
 if lock.exists():require(readj(lock)==state,'config/code changed; use a new output run directory')
 else:writej(lock,state)
 refpaths=[expand(c,c[k]) for k in ['six_metrics','behavior_table','nine_edges']] + [expand(c,c['original_input'][m],s) for s in SUBJECTS for m in MODELS] + [expand(c,c['original_output'][m],s) for s in c['pilot_subjects'] for m in MODELS]
 refs=[fingerprint(p,True) for p in refpaths]; rf=out/'reference_fingerprints.json'
 if rf.exists():require(readj(rf)==refs,'reference inputs changed; new run required')
 else:writej(rf,refs)
 reference_checks(c,out)
 if a.mode=='prepare':prepare(c,out)
 else:
  if a.mode=='formal':
   pp=out/'pilot_report.json';require(pp.is_file() and readj(pp)['status']=='PASS','pilot must pass');require(a.pilot_review_sha==sha(pp),'formal requires reviewed pilot_report SHA256; do not select based on effects')
  execute(c,out,a.mode,a.retry_failed)
if __name__=='__main__':main()
