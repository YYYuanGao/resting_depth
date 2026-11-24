"""Read-only QC of existing pial5 parcel composites; no MRI or rDCM fitting.
Python 3.8+, NumPy and SciPy. Each invocation writes a NEW output directory.
"""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import time
import zipfile
from pathlib import Path
import numpy as np
import scipy
from scipy.io import loadmat


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def table(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def correlation(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a-a.mean();b=b-b.mean();norm=np.linalg.norm(a)*np.linalg.norm(b)
    return None if norm==0 else float(np.clip(np.dot(a,b)/norm,-1,1))


def cell_metrics(x,names):
    """Metrics on stored parcel signals; center for waveform differences only.
    No parcel z-scoring, temporal resampling, or deletion before aggregation.
    """
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)!=len(names) or len(names)<2 or not np.isfinite(x).all():
        raise ValueError('Expected at least two finite parcel time series')
    m,t=x.shape;full=x.mean(axis=0);centered=full-full.mean();sd=float(full.std())
    if sd==0:raise ValueError('Zero-variance full composite')
    pairs=[];removed=[]
    for i,j in itertools.combinations(range(m),2):
        pairs.append(dict(parcel_a=names[i],parcel_b=names[j],pearson_r=correlation(x[i],x[j])))
    for i,p in enumerate(names):
        loo=np.delete(x,i,axis=0).mean(axis=0)
        difference=(loo-loo.mean())-centered
        nr=float(np.sqrt(np.mean(difference**2))/sd)
        # Signed contributions sum to one, and may be negative or exceed one.
        contribution=float(np.mean((x[i]-x[i].mean())*centered)/(m*sd**2))
        loo_r=correlation(loo,full);rest_r=correlation(x[i],loo)
        removed.append(dict(parcel=p,n_parcels=m,n_timepoints=t,
            parcel_mean=float(x[i].mean()),parcel_sd=float(x[i].std()),
            parcel_to_rest_r=rest_r,loo_vs_full_r=loo_r,
            loo_centered_rmse_over_full_sd=nr,
            count_adjusted_change=(m-1)*nr,
            loo_sd_over_full_sd=float(loo.std()/sd),
            loo_mean_shift_over_full_sd=float((loo.mean()-full.mean())/sd),
            variance_contribution_signed=contribution,
            status='OK' if loo_r is not None and rest_r is not None else 'UNDEFINED_CORRELATION',
            reason='' if loo_r is not None and rest_r is not None else 'A removed/composite signal has zero centered variance'))
    rs=[r['pearson_r'] for r in pairs if r['pearson_r'] is not None]
    rest=[r['parcel_to_rest_r'] for r in removed if r['parcel_to_rest_r'] is not None]
    lr=[r['loo_vs_full_r'] for r in removed if r['loo_vs_full_r'] is not None]
    worst=max(removed,key=lambda r:r['loo_centered_rmse_over_full_sd'])
    summary=dict(n_parcels=m,n_timepoints=t,full_mean=float(full.mean()),full_sd=sd,
        pairwise_r_median=float(np.median(rs)) if rs else None,
        pairwise_r_min=min(rs) if rs else None,n_pairs=len(pairs),n_pairs_defined=len(rs),
        fraction_negative_pairs=float(np.mean(np.array(rs)<0)) if rs else None,
        parcel_to_rest_r_min=min(rest) if rest else None,loo_vs_full_r_min=min(lr) if lr else None,
        max_loo_centered_rmse_over_full_sd=worst['loo_centered_rmse_over_full_sd'],
        max_count_adjusted_change=worst['count_adjusted_change'],worst_removed_parcel=worst['parcel'],
        n_undefined_loo=sum(r['status']!='OK' for r in removed))
    return pairs,removed,summary


def subject_paths(cfg,subject):
    values=dict(project_root=cfg['project_root'],subject=subject)
    folder=Path(cfg['parcel_directory'].format(**values)).expanduser().resolve()
    return folder,{k:Path(v.format(**values)).expanduser().resolve() for k,v in cfg['reference_inputs'].items()}


def validate_config(cfg):
    ids=cfg['subjects'];members=cfg['members']
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Empty or duplicate subject list')
    if cfg['depths']!=['deep','mid','sup']:raise ValueError('Depth order must match reviewed source: deep/mid/sup')
    flat=[p for names in members.values() for p in names]
    if len(flat)!=len(set(flat)) or any(len(v)<2 for v in members.values()):
        raise ValueError('Overlapping stage memberships or fewer than two parcels')
    if set(members)!={'EVC','PPC','cauPFC'}:raise ValueError('Expected three frozen stages')
    if cfg['reference_stages']!={'six':['EVC','cauPFC'],'nine':['EVC','PPC','cauPFC']}:
        raise ValueError('Unexpected six/nine node order')
    if cfg['expected_timepoints']<3 or cfg['min_std']<=0:raise ValueError('Invalid signal constraints')
    for tol in cfg['parity_tolerance'].values():
        if not np.isfinite(tol) or tol<0:raise ValueError('Invalid parity tolerance')


def checked_file(path,subject,role,cfg,files):
    if not path.is_file():raise FileNotFoundError(str(path))
    size=path.stat().st_size
    if size>cfg['max_file_bytes']:raise ValueError('Input exceeds configured small-file limit: '+str(path))
    files.append(dict(subject=subject,role=role,path=str(path),bytes=size,sha256=sha(path)))
    return path


def canonical(value,aliases):
    if isinstance(value,bytes):value=value.decode('utf-8')
    if not isinstance(value,(str,np.str_)):raise ValueError('ROI name is not a string')
    value=str(value).strip()
    if not value:raise ValueError('Empty ROI name')
    return aliases.get(value,value)


def matlab_labels(value):
    out=[]
    for v in np.asarray(value,dtype=object).ravel():
        while isinstance(v,np.ndarray) and v.size==1:v=v.item()
        if not isinstance(v,str):raise ValueError('Unknown MATLAB label representation')
        out.append(v.strip())
    return out


def inspect_subject(cfg,subject,files,checks,parity):
    folder,refs=subject_paths(cfg,subject)
    fp=checked_file(folder/'ts_matrix.npy',subject,'parcel_timeseries',cfg,files)
    ts=np.load(fp,allow_pickle=False)
    if ts.dtype.kind!='f' or ts.ndim!=3 or ts.shape[1:]!=(3,cfg['expected_timepoints']):
        raise ValueError('Unexpected floating ROI x depth x time array: '+str(ts.shape))
    rp=checked_file(folder/'roi_names.npy',subject,'roi_names',cfg,files)
    names_raw=np.load(rp,allow_pickle=bool(cfg['trusted_roi_name_pickle']))
    if names_raw.ndim!=1:raise ValueError('ROI names must be one-dimensional')
    names=[canonical(v,cfg.get('roi_aliases',{})) for v in names_raw]
    if len(names)!=ts.shape[0] or len(names)!=len(set(names)):raise ValueError('ROI count or canonical duplicates')
    mp=checked_file(folder/'pial_exclusion_metadata.json',subject,'pial_metadata',cfg,files)
    meta=json.loads(mp.read_text())
    for k,v in [('subject',subject),('n_rois',len(names)),('n_layers',3),('n_timepoints',ts.shape[2])]:
        if meta.get(k)!=v:raise ValueError('Extraction metadata mismatch: '+k)
    lookup={n:i for i,n in enumerate(names)};selected={};composites={}
    for stage,parcels in cfg['members'].items():
        missing=[p for p in parcels if p not in lookup]
        if missing:raise ValueError(stage+' missing parcels: '+str(missing))
        idx=[lookup[p] for p in parcels];x=ts[idx]
        if not np.isfinite(x).all() or np.any(x.std(axis=2)<cfg['min_std']):
            raise ValueError(stage+' has nonfinite/low-variance selected parcel signals')
        selected[stage]=x
        # Match the exporter's original float dtype and operation order exactly.
        composites[stage]=np.stack([x[:,di,:].mean(axis=0) for di in range(3)],axis=1).astype(float)
        if np.any(composites[stage].std(axis=0)<cfg['min_std']):raise ValueError(stage+' low-variance full composite')
    counts=None;cp=folder/'n_voxels_clean.npy'
    if cp.exists():
        checked_file(cp,subject,'voxel_counts',cfg,files);counts=np.load(cp,allow_pickle=False)
        if counts.shape!=ts.shape[:2] or counts.dtype.kind not in 'iuf' or not np.isfinite(counts).all() or np.any(counts<0) or not np.equal(counts,np.rint(counts)).all():
            raise ValueError('Invalid parcel voxel-count array')
        if any(np.any(counts[[lookup[p] for p in ps]]<=0) for ps in cfg['members'].values()):
            raise ValueError('Nonpositive selected-parcel count')
    else:checks.append(dict(subject=subject,check='voxel_counts',status='MISSING_OPTIONAL',reason=str(cp)))
    failures=[]
    for model,path in refs.items():
        checked_file(path,subject,'original_'+model+'_input',cfg,files)
        mat=loadmat(path,squeeze_me=True,struct_as_record=False)
        expected=cfg['reference_labels'][model]
        if matlab_labels(mat.get('labels',[]))!=expected:raise ValueError(model+' node label/order mismatch')
        if str(mat.get('Subject',''))!=subject:raise ValueError(model+' Subject metadata mismatch')
        reference=np.asarray(mat.get('ts'),dtype=float)
        rebuilt=np.column_stack([composites[s] for s in cfg['reference_stages'][model]])
        if reference.shape!=rebuilt.shape or not np.isfinite(reference).all():raise ValueError(model+' unexpected reference ts')
        if not np.isclose(float(mat.get('TR',0)),2.2) or not np.isclose(float(mat.get('TE',0)),.026):
            raise ValueError(model+' TR/TE differs from reviewed branch')
        if int(mat.get('n_nodes',0))!=len(expected):raise ValueError(model+' n_nodes mismatch')
        if model=='nine':
            saved=json.loads(str(mat.get('stage_members_json','{}')))
            if saved!=cfg['members']:raise ValueError('Saved nine-node stage membership/order differs from config')
        else:
            if str(mat.get('validation_branch',''))!='pial5_current':raise ValueError('Six-node branch mismatch')
        tol=cfg['parity_tolerance']
        for col,label in enumerate(expected):
            err=rebuilt[:,col]-reference[:,col];refsd=float(reference[:,col].std())
            normalized=float(np.sqrt(np.mean(err**2))/refsd) if refsd>0 else None
            ok=normalized is not None and normalized<=tol['rmse_over_sd'] and np.allclose(rebuilt[:,col],reference[:,col],rtol=tol['rtol'],atol=tol['atol'])
            parity.append(dict(subject=subject,model=model,node=label,max_abs_error=float(np.max(np.abs(err))),rmse_over_reference_sd=normalized,status='PASS' if ok else 'FAIL'))
            if not ok:failures.append(model+':'+label)
    if failures:raise ValueError('Reconstruction parity failed: '+','.join(failures))
    checks.append(dict(subject=subject,check='input_and_parity',status='PASS',reason='All selected parcels and both model inputs verified; '+str(ts.shape)))
    return selected,counts,lookup


def run(cfg,out,inspect_only=False):
    validate_config(cfg);out=Path(out).expanduser().resolve()
    for s in cfg['subjects']:
        folder,refs=subject_paths(cfg,s)
        for protected in [folder]+[p.parent for p in refs.values()]:
            if out==protected or out in protected.parents or protected in out.parents:
                raise ValueError('Output overlaps an input directory: '+str(protected))
    if out.exists():raise FileExistsError('Use a NEW output directory: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
    dump(out/'config_used.json',cfg)
    files=[];checks=[];parity=[];pairs=[];removed=[];cells=[];start=time.time()
    subjects=[]
    for subject in cfg['subjects']:
        started=time.time()
        try:
            selected,counts,lookup=inspect_subject(cfg,subject,files,checks,parity)
            pp=[];rr=[];cc=[]
            if not inspect_only:
                for stage,names in cfg['members'].items():
                    for di,depth in enumerate(cfg['depths']):
                        common=dict(subject=subject,stage=stage,depth=depth)
                        p,r,c=cell_metrics(selected[stage][:,di,:],names)
                        pp.extend(dict(**common,**item) for item in p)
                        for item in r:
                            item['n_voxels']=int(counts[lookup[item['parcel']],di]) if counts is not None else None
                            rr.append(dict(**common,**item))
                        cc.append(dict(**common,**c))
                pairs.extend(pp);removed.extend(rr);cells.extend(cc)
            subjects.append(dict(subject=subject,status='COMPLETE',reason='',elapsed_seconds=time.time()-started))
            print(subject+': input parity passed'+('; QC computed' if not inspect_only else ''),flush=True)
        except Exception as e:
            reason=type(e).__name__+': '+str(e)
            checks.append(dict(subject=subject,check='subject_validation',status='FAIL',reason=reason))
            subjects.append(dict(subject=subject,status='BLOCKED',reason=reason,elapsed_seconds=time.time()-started))
            print(subject+': BLOCKED '+reason,flush=True)
    tables={'checks.csv':(checks,['subject','check','status','reason']),
            'input_files.csv':(files,['subject','role','path','bytes','sha256']),
            'node_reconstruction.csv':(parity,['subject','model','node','max_abs_error','rmse_over_reference_sd','status']),
            'pairwise_correlations.csv':(pairs,['subject','stage','depth','parcel_a','parcel_b','pearson_r']),
            'parcel_leave_one_out.csv':(removed,['subject','stage','depth','parcel','n_parcels','n_timepoints','parcel_mean','parcel_sd','parcel_to_rest_r','loo_vs_full_r','loo_centered_rmse_over_full_sd','count_adjusted_change','loo_sd_over_full_sd','loo_mean_shift_over_full_sd','variance_contribution_signed','status','reason','n_voxels']),
            'subject_stage_depth.csv':(cells,['subject','stage','depth','n_parcels','n_timepoints','full_mean','full_sd','pairwise_r_median','pairwise_r_min','n_pairs','n_pairs_defined','fraction_negative_pairs','parcel_to_rest_r_min','loo_vs_full_r_min','max_loo_centered_rmse_over_full_sd','max_count_adjusted_change','worst_removed_parcel','n_undefined_loo'])}
    for name,(data,fields) in tables.items():table(out/name,data,fields)
    summary=[]
    for stage in cfg['members']:
        for depth in cfg['depths']:
            rows=[r for r in cells if r['stage']==stage and r['depth']==depth]
            if not rows:continue
            base=dict(stage=stage,depth=depth,n_subjects=len(rows),expected_subjects=len(cfg['subjects']))
            for k in ['pairwise_r_median','parcel_to_rest_r_min','loo_vs_full_r_min','max_loo_centered_rmse_over_full_sd','max_count_adjusted_change']:
                vals=[r[k] for r in rows if r[k] is not None]
                summary.append(dict(**base,metric=k,n_defined=len(vals),median=float(np.median(vals)) if vals else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None))
    table(out/'group_descriptive_summary.csv',summary,['stage','depth','n_subjects','expected_subjects','metric','n_defined','median','minimum','maximum'])
    complete=sum(r['status']=='COMPLETE' for r in subjects)
    status=dict(execution_status='COMPLETE' if complete==len(subjects) else 'INCOMPLETE',
        scientific_status='DESCRIPTIVE_QC_REQUIRES_REVIEW; not proof of valid grouping or robust rDCM',
        mode='inspect_only' if inspect_only else 'qc',n_subjects_requested=len(subjects),n_subjects_complete=complete,
        subjects=subjects,n_cells=len(cells),n_pairwise_rows=len(pairs),n_loo_rows=len(removed),
        n_undefined_loo=sum(r['status']!='OK' for r in removed),elapsed_seconds=time.time()-start,
        program_sha256=sha(Path(__file__).resolve()),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        tsnr='NOT_COMPUTED: appropriate mean baseline not established',inference='No p values; no temporal or parcel pseudo-replication')
    dump(out/'status.json',status)
    lines=['# Parcel composite QC', '', 'Execution: '+status['execution_status'],
           'Participants complete: %d/%d'%(complete,len(subjects)),
           'This is signal-level descriptive QC, not a test of rDCM parameter robustness.',
           'No MRI was read, no rDCM fitted, no parcels removed from the analysis.',
           'Undefined correlations are blank in CSV; inspect status.json and parcel_leave_one_out.csv.',
           '', '|Stage|Depth|Metric|N defined|Median|Min|Max|','|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda v:'NA' if v is None else '%.5f'%v
        lines.append('|%s|%s|%s|%d|%s|%s|%s|'%(r['stage'],r['depth'],r['metric'],r['n_defined'],fmt(r['median']),fmt(r['minimum']),fmt(r['maximum'])))
    lines.extend(['','## Interpretation',
        '- parcel_to_rest_r excludes the parcel from its comparison composite.',
        '- loo_vs_full_r shares most parcels and is mechanically high, especially for larger groups.',
        '- count_adjusted_change = (m-1) * centered LOO RMSE / full SD; descriptive, not a validated cutoff.',
        '- Equal parcel weights do not guarantee equal signal variance contributions.',
        '- Review stage/depth differences and per-parcel patterns. Do not choose a grouping by significance.',
        '- No generic PASS cutoff is assigned to correlations. High coherence can reflect shared nuisance signal.',
        '- Source signals are stored pre-rDCM-export parcel means; original demeaning/global model scaling is not altered.',
        '- Per-timepoint samples are autocorrelated. No naive correlation significance tests are performed.'])
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Package only reports, never source arrays, MRI, or original model files.
    archive=out/'parcel_qc_return.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.csv','.json','.md']:z.write(p,out.name+'/'+p.name)
    print('Return: '+str(archive),flush=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--project-root',help='Optional override of the verified root')
    p.add_argument('--inspect-only',action='store_true',help='Input/schema/parity checks only; no QC metrics')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.project_root:cfg['project_root']=a.project_root
    status=run(cfg,a.out,a.inspect_only)
    raise SystemExit(0 if status['execution_status']=='COMPLETE' else 2)


if __name__=='__main__':main()
