"""Participant-level inference only; repetitions never treated as participants."""
import itertools
import numpy as np
from scipy import stats
from pathlib import Path
from roi_size import csvread,csvwrite,writej,expand,require,SUBJECTS

def holm(p):
 p=np.asarray(p,float);order=np.argsort(p);a=np.empty(len(p));a[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return a

def mean_test(x):
 x=np.asarray(x,float);n=len(x);sd=x.std(ddof=1);se=sd/np.sqrt(n)
 ci=(float(x.mean()),float(x.mean())) if not se else stats.t.interval(.95,n-1,loc=x.mean(),scale=se)
 signs=np.array(list(itertools.product([-1,1],repeat=n)))
 return dict(n=n,mean=x.mean(),sd=sd,ci_low=ci[0],ci_high=ci[1],t=float(stats.ttest_1samp(x,0).statistic),p_t=float(stats.ttest_1samp(x,0).pvalue),cohen_dz=x.mean()/sd if sd else np.nan,p_signflip=float(np.mean(np.abs(signs@x/n)>=abs(x.mean())-1e-15)))

def spearman(a,b):return float(stats.spearmanr(a,b).statistic)
def behavior_test(x,y,old,nboot,nperm,rng):
 require(all(np.isfinite(v).all() and np.ptp(v)>0 for v in [x,y,old]),'Nonfinite or constant participant vector: correlation inference blocked')
 obs=spearman(x,y);oldrho=spearman(old,y);bs=[];delta=[]
 for _ in range(nboot):
  k=rng.integers(0,len(x),len(x));r=spearman(x[k],y[k]);o=spearman(old[k],y[k])
  if np.isfinite(r) and np.isfinite(o):bs.append(r);delta.append(r-o)
 rx=stats.rankdata(x);ry=stats.rankdata(y);rx=(rx-rx.mean())/np.linalg.norm(rx-rx.mean());ry=(ry-ry.mean())/np.linalg.norm(ry-ry.mean());exceed=0
 for _ in range(nperm):exceed+=abs(rx@rng.permutation(ry))>=abs(obs)-1e-15
 require(len(bs)>0,'No valid paired bootstrap draws')
 return dict(rho=obs,original_rho=oldrho,delta_rho=obs-oldrho,ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975),paired_delta_ci_low=np.quantile(delta,.025),paired_delta_ci_high=np.quantile(delta,.975),bootstrap_valid=len(bs),bootstrap_requested=nboot,p_permutation=(exceed+1)/(nperm+1),n_permutations=nperm)

def summarize(c,out):
 fits=csvread(out/'formal_fits.csv');nrep=c['formal_repeats'];expected={(s,m,r) for s in SUBJECTS for m in ['six','nine'] for r in range(nrep)}
 require({(r['subject'],r['model'],int(r['rep'])) for r in fits}==expected and len(fits)==len(expected) and all(r['status']=='SUCCESS' for r in fits),'Incomplete fits: no group inference; retry same failed tasks')
 b=csvread(expand(c,c['behavior_table']));require({r['subject'] for r in b}==set(SUBJECTS) and len(b)==15,'Behavior ID mismatch')
 b={r['subject']:r for r in b};six=csvread(expand(c,c['six_metrics']));old6={r['subject']:r for r in six if r['network']=='cauPFC_EVC' and r['condition']=='pial5_current'};v1={r['subject']:r for r in six if r['network']=='cauPFC_V1' and r['condition']=='pial5_current'}
 require(set(old6)==set(v1)==set(SUBJECTS),'Original six-node metrics incomplete')
 participant=[];matched={};ranges=[]
 for m in ['six','nine']:
  for s in SUBJECTS:
   a=sorted([r for r in fits if r['subject']==s and r['model']==m],key=lambda r:int(r['rep']));ff=np.array([float(r['FF']) for r in a]);fb=np.array([float(r['FB']) for r in a]);row=dict(subject=s,model=m,n_repeats=nrep,FF=float(ff.mean()),FB=float(fb.mean()),H=float(fb.mean()-ff.mean()))
   for k,val in [('FF',ff),('FB',fb),('H',fb-ff)]:
    orig=float((old6 if m=='six' else b)[s]['signed_'+k]) if k!='H' or m=='six' else float(b[s]['signed_FB'])-float(b[s]['signed_FF'])
    row['original_'+k]=orig;row['delta_'+k]=row[k]-orig
    ranges.append(dict(subject=s,model=m,metric=k,min=float(val.min()),max=float(val.max()),sd=float(val.std(ddof=1)) if nrep>1 else 0,interpretation='voxel_sampling_range_not_participant_CI'))
   participant.append(row);matched[s,m]=row
 csvwrite(out/'participant_summary.csv',participant);csvwrite(out/'sampling_ranges.csv',ranges)
 group=[];paired=[]
 for k in ['FF','FB','H']:
  x=np.array([matched[s,'six'][k] for s in SUBJECTS]);o=np.array([float(old6[s]['signed_'+k]) for s in SUBJECTS]);vx=np.array([float(v1[s]['signed_'+k]) for s in SUBJECTS]);vt=mean_test(vx)
  for condition,a in [('matched',x),('original',o)]:
   r=mean_test(a);r.update(model='six',metric=k,condition=condition,p_t_holm_original_two_network_family=float(holm([r['p_t'],vt['p_t']])[0]),p_signflip_holm_original_two_network_family=float(holm([r['p_signflip'],vt['p_signflip']])[0]),family_note='EVC current estimate plus fixed original unmatched V1 auxiliary; not Holm across FF/FB/H',V1_fixed_p_t=vt['p_t'],V1_fixed_p_signflip=vt['p_signflip']);group.append(r)
  paired.append(dict(metric=k,model='six',contrast='matched_minus_original',**mean_test(x-o)))
 csvwrite(out/'six_group_statistics.csv',group);csvwrite(out/'six_paired_changes.csv',paired)
 rng=np.random.default_rng(c['statistics_seed']);bh=[]
 for outcome in ['Threshold','ACC_all']:
  for k in ['FB','FF']:
   x=np.array([matched[s,'nine'][k] for s in SUBJECTS]);y=np.array([float(b[s][outcome]) for s in SUBJECTS]);o=np.array([float(b[s]['signed_'+k]) for s in SUBJECTS]);bh.append(dict(outcome=outcome,metric=k,model='nine',**behavior_test(x,y,o,c['behavior_bootstrap'],c['behavior_permutations'],rng)))
 for outcome in ['Threshold','ACC_all']:
  a=[r for r in bh if r['outcome']==outcome]
  for r,p in zip(a,holm([r['p_permutation'] for r in a])):r['p_holm_within_outcome_two_pathways']=p
 csvwrite(out/'nine_behavior_statistics.csv',bh)
 repetitions=[]
 for rep in range(nrep):
  a={(r['subject'],r['model']):r for r in fits if int(r['rep'])==rep}
  for k in ['FF','FB','H']:repetitions.append(dict(rep=rep,model='six',metric=k,value=np.mean([float(a[s,'six'][k]) for s in SUBJECTS]),interpretation='descriptive_sampling_sensitivity_not_formal_inference'))
  for outcome in ['Threshold','ACC_all']:
   for k in ['FB','FF']:repetitions.append(dict(rep=rep,model='nine',metric=k,outcome=outcome,value=spearman([float(a[s,'nine'][k]) for s in SUBJECTS],[float(b[s][outcome]) for s in SUBJECTS]),interpretation='descriptive_sampling_sensitivity_not_formal_inference'))
 csvwrite(out/'repeat_descriptive_results.csv',repetitions)
 repeat_ranges=[]
 for key in sorted({(r['model'],r['metric'],r.get('outcome','')) for r in repetitions}):
  vals=np.array([r['value'] for r in repetitions if (r['model'],r['metric'],r.get('outcome',''))==key]);finite=vals[np.isfinite(vals)]
  repeat_ranges.append(dict(model=key[0],metric=key[1],outcome=key[2],min=float(finite.min()) if len(finite) else None,max=float(finite.max()) if len(finite) else None,n_valid=len(finite),n_repeats=len(vals),interpretation='sampling_range_not_participant_CI_no_significance_proportion'))
 csvwrite(out/'repeat_descriptive_ranges.csv',repeat_ranges)
 writej(out/'statistics_status.json',dict(status='COMPLETE',n_participants=15,n_repeats=nrep,unit='participant; mean connectivity across repetitions before H/correlation',paired_change_tests='six paired-change p values are unadjusted descriptive diagnostics, not additional primary claims',limitations='Voxel-count sensitivity only; not independent information/SNR matching; anatomical and vascular differences remain'))
