"""Run: python test_roi_size.py. Synthetic only; no MATLAB/server claims."""
import json,sys,tempfile,types
from pathlib import Path
import numpy as np
from scipy.io import savemat,loadmat
import roi_size as r
from summarize_roi_size import holm,mean_test,behavior_test,summarize

def main():
 rng=np.random.default_rng(3)
 # Exhaustive small budgets versus independently enumerated feasible quotas.
 for _ in range(100):
  caps={('EVC','deep'):rng.integers(5,15,3).tolist(),('cauPFC','deep'):rng.integers(5,15,7).tolist()}
  possible=[]
  for b in range(1,min(map(sum,caps.values()))+1):
   valid=True
   for v in caps.values():
    q,n=divmod(b,len(v));valid=valid and q>=1 and all(x>=q for x in v) and sum(x>=q+1 for x in v)>=n
   if valid:possible.append(b)
  B,qq,_=r.allocations(caps,1);assert B==max(possible)
  for k,a in qq.items():assert sum(a)==B and max(a)-min(a)<=1 and all(x<=y for x,y in zip(a,caps[k]))
 for L in range(0,12):
  for R in range(0,12):
   for q in range(max(1,int(L>0)+int(R>0)),L+R+1):
    l,rr,ideal=r.hemi_quota(q,L,R);assert l+rr==q and l<=L and rr<=R and (not L or l>0) and (not R or rr>0) and abs(l-ideal)<=1
 assert np.allclose(holm([.01,.03,.04]),[.03,.06,.06])
 try:behavior_test(np.ones(15),np.arange(15),np.arange(15),10,10,rng);raise AssertionError('constant correlation accepted')
 except ValueError:pass
 try:r.allocations({('EVC','d'):[5]*3,('PPC','d'):[5]*13},5);raise AssertionError('low budget accepted')
 except ValueError:pass
 with tempfile.TemporaryDirectory() as td:
  root=Path(td).resolve();td=str(root);c=json.loads(Path(__file__).with_name('config.json').read_text());c.update(project_root=td,step3=td+'/data/{subject}',output=td+'/roi_size_sensitivity/run',hemisphere_provenance='synthetic 1=L 2=R',hash_large_inputs=False)
  s='sub-01';base=root/'data'/s;(base/'clean_masks').mkdir(parents=True);names=[v for a in r.STAGES.values() for v in a];shape=(23,3,60)
  data=(100+10*np.sin(np.arange(218)/12)[None,None,None,:]+rng.normal(size=shape+(218,))).astype(np.float32)
  ts=data.mean(axis=2,dtype=np.float32);np.save(base/'roi_names.npy',np.array(names));np.save(base/'ts_matrix.npy',ts);np.save(base/'n_voxels_clean.npy',np.full((23,3),60))
  bold=root/'bold.nii';hemi=root/'hemi.nii';bold.write_bytes(b'fixture');hemi.write_bytes(b'fixture');r.writej(base/'pial_exclusion_metadata.json',{'bold_file':str(bold)})
  c['hemisphere_images'][s]=str(hemi);objects={str(bold):data,str(hemi):np.broadcast_to(np.where(np.arange(60)<30,1,2),shape)}
  for i,d in enumerate(r.LAYERS):
   path=base/'clean_masks'/f'layer{i+1}_clean.nii.gz';path.write_bytes(b'fixture');mask=np.zeros(shape,int);mask[:,i,:]=np.arange(1,24)[:,None];objects[str(path)]=mask
  # Mock only the NIfTI decoding interface; actual nibabel/grid IO remains a server test.
  fake=types.SimpleNamespace(load=lambda p,**kwargs:types.SimpleNamespace(shape=objects[str(p)].shape,affine=np.eye(4),dataobj=objects[str(p)]));saved=sys.modules.get('nibabel');sys.modules['nibabel']=fake
  for m in r.MODELS:
   path=root/f'{m}.mat';c['original_input'][m]=str(path);savemat(path,dict(ts=r.nodes(ts,names,m),labels=np.array(r.labels(m),object),TR=2.2,TE=.026))
  out=root/'roi_size_sensitivity/run';cache=out/'cache'/s
  counts,parity,node=r.read_subject(c,s,cache);assert len(counts)==138 and all(x['pass_check'] for x in parity+node)
  plan,budgets=r.plan_subject(c,s,counts);assert {x['node_budget'] for x in budgets}=={180}
  r.csvwrite(out/'sampling_plan.csv',plan);c['pilot_subjects']=[s]
  jobs=r.make_jobs(c,out,'pilot');assert len(jobs)==6
  assert jobs==r.make_jobs(c,out,'pilot')
  for j in jobs:
   z=np.load(Path(j['job_dir'])/'sample_indices.npz');chosen=np.concatenate([z[k] for k in z]);assert len(chosen)==len(set(chosen));inp=loadmat(j['input'])['ts'];assert inp.shape==(218,6 if j['model']=='six' else 9)
   recreated=ts.copy()
   for seg in r.readj(cache/'source.json')['segments']:
    if seg['stage'] not in r.MODELS[j['model']]:continue
    ix=np.sort(np.concatenate([z[f'{seg["parcel"]}__{seg["depth"]}__{h}'] for h in ['L','R']]))
    recreated[names.index(seg['parcel']),r.LAYERS.index(seg['depth'])]=data.reshape(-1,218)[ix].mean(axis=0,dtype=np.float32)
   assert np.array_equal(inp,r.nodes(recreated,names,j['model']))
  # Grid/depth-overlap failure must block before sampling.
  original_mask=objects[str(base/'clean_masks/layer1_clean.nii.gz')].copy()
  objects[str(base/'clean_masks/layer1_clean.nii.gz')]=objects[str(base/'clean_masks/layer2_clean.nii.gz')]
  try:r.read_subject(c,s,out/'overlap');raise AssertionError('overlap accepted')
  except ValueError:pass
  objects[str(base/'clean_masks/layer1_clean.nii.gz')]=original_mask
  assert not r.compare(np.eye(6),np.eye(6)+.1,c,A=True)['pass_check']
  # Wrong original baseline blocks preparation rather than silently accepting.
  np.save(base/'ts_matrix.npy',ts+1)
  try:r.read_subject(c,s,out/'bad');raise AssertionError('bad baseline accepted')
  except ValueError:pass
  if saved is None:sys.modules.pop('nibabel',None)
  else:sys.modules['nibabel']=saved
  # Participant means, all four behavior comparisons, paired change bootstrap.
  fits=[];old=[];beh=[];c['formal_repeats']=3;c['behavior_bootstrap']=25;c['behavior_permutations']=50
  for i,s in enumerate(r.SUBJECTS):
   ff=-.03+i*.001;fb=.2-i*.004
   for net in ['cauPFC_EVC','cauPFC_V1']:old.append(dict(subject=s,network=net,condition='pial5_current',signed_FF=ff,signed_FB=fb,signed_H=fb-ff))
   beh.append(dict(subject=s,signed_FF=ff,signed_FB=fb,Threshold=i+1,ACC_all=(i*7)%15))
   for m in r.MODELS:
    for rep in range(3):fits.append(dict(subject=s,model=m,rep=rep,status='SUCCESS',FF=ff+(rep-1)*.001,FB=fb+(rep-1)*.002,H=fb-ff+(rep-1)*.001))
  r.csvwrite(out/'formal_fits.csv',fits);r.csvwrite(root/'six.csv',old);r.csvwrite(root/'beh.csv',beh);c['six_metrics']=str(root/'six.csv');c['behavior_table']=str(root/'beh.csv');summarize(c,out)
  a=r.csvread(out/'participant_summary.csv');assert len(a)==30 and all(abs(float(x['delta_FF']))<1e-12 for x in a)
  assert len(r.csvread(out/'nine_behavior_statistics.csv'))==4
  fits.pop();r.csvwrite(out/'formal_fits.csv',fits)
  try:summarize(c,out);raise AssertionError('missing repeat accepted')
  except ValueError:pass
 print('PASS: exact budget maxima, hemi coverage/rounding, low-budget block, synthetic all-voxel reconstruction, saved-index reproducibility, bad-baseline block, participant aggregation, four behavior tests, incomplete-fit block. NIfTI decoding and MATLAB fitting not tested locally.')
if __name__=='__main__':main()
