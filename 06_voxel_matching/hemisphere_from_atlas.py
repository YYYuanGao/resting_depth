#!/usr/bin/env python3
"""Check existing atlas hemisphere coverage; optionally write exact-label maps.
Never resamples, dilates, fills unknown voxels or executes the source builder.
"""
import argparse, ast, json
from pathlib import Path
import numpy as np
from roi_size import readj,writej,csvwrite,sha,expand,sources,STAGES,SUBJECTS,LAYERS,require

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh

BUILDER_SHA='d396382034ae81579055c3aa1e37ea0a5c34eaf3523b178d0c38ca6e81fa4114'
PRIMARY=PROJECT_ROOT + '/Atlas_HCPMMP/func_space_MPRAGEisedAffine/{subject}_HCPMMP1_in_func.nii.gz'
SECONDARY='{project_root}/S1_Anat2Func/hcpmmp2func/{subject}_HCPMMP1_in_func.nii.gz'

def lookup(builder):
 require(sha(builder)==BUILDER_SHA,'Builder version differs from audited source; review label LUT before assigning hemispheres')
 tree=ast.parse(Path(builder).read_text())
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HCP_MMP_LH_LABELS' for t in node.targets):
   left=ast.literal_eval(node.value);require(len(left)==179,'Unexpected LUT length');return left
 raise ValueError('HCP_MMP_LH_LABELS absent')

def classify(atlas,left):
 a=np.asarray(atlas);require(np.isfinite(a).all() and np.allclose(a,np.rint(a),atol=1e-5,rtol=0),'Atlas must have finite integer labels')
 a=np.rint(a).astype(np.int32);lh=list(left.values());rh=[v+1000 for v in lh]
 require(not set(lh)&set(rh),'Overlapping LH/RH codes')
 h=np.zeros(a.shape,np.uint8);h[np.isin(a,lh)]=1;h[np.isin(a,rh)]=2;return h

def assess(c,s,atlas_path,left):
 import nibabel as nib
 src=sources(c,s)
 require(atlas_path.is_file(),f'Atlas not found: {atlas_path}')
 require(src['bold'] is not None and src['bold'].is_file(),'Original BOLD path unavailable; fill bold_override or restore pial metadata')
 ref=nib.load(src['bold']);require(len(ref.shape)==4,'BOLD must be 4D')
 def load3(p):
  img=nib.load(p);require(img.shape==ref.shape[:3] and np.allclose(img.affine,ref.affine,atol=1e-4,rtol=0),f'Grid/affine mismatch: {p}')
  a=np.asarray(img.dataobj);require(np.isfinite(a).all() and np.allclose(a,np.rint(a),atol=1e-5,rtol=0),f'Invalid label values: {p}')
  return img,np.rint(a).astype(np.int32)
 atlas_img,atlas=load3(atlas_path);hemi=classify(atlas,left)
 names=np.load(src['roi_names'],allow_pickle=c['trusted_roi_name_pickle']).astype(str)
 names=[c['roi_aliases'].get(x,x) for x in names];require(len(names)==len(set(names)),'Duplicate ROI names')
 require(set(sum(STAGES.values(),[]))<=set(names),'Frozen parcels missing')
 counts=np.load(src['counts'],allow_pickle=False);require(counts.shape==(len(names),3),'Count dimensions')
 seen=np.zeros(ref.shape[:3],bool);rows=[]
 for d in LAYERS:
  _,mask=load3(src[d]);require(mask.min()>=0 and mask.max()<=len(names),'Final mask index range')
  require(not np.any(seen & (mask>0)),'Final depth masks overlap');seen|=mask>0
  for stage,rois in STAGES.items():
   for roi in rois:
    i=names.index(roi);sel=mask==i+1;n=int(sel.sum());require(n==int(counts[i,LAYERS.index(d)]),'Final mask/count disagreement')
    require(n>=5,'Required parcel has fewer than five retained voxels')
    l=int(np.sum(hemi[sel]==1));r=int(np.sum(hemi[sel]==2));unknown=n-l-r
    mismatch=int(np.sum((hemi[sel]>0) & ~np.isin(atlas[sel],[left[roi],left[roi]+1000])))
    rows.append(dict(subject=s,stage=stage,parcel=roi,depth=d,n_final=n,n_L=l,n_R=r,n_unknown=unknown,n_other_parcel_atlas_label=mismatch,atlas_path=str(atlas_path)))
 return atlas_img,hemi,rows,dict(atlas_sha256=sha(atlas_path),bold_header_reference=str(src['bold']),shape=list(ref.shape[:3]),affine=ref.affine.tolist())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config.json');p.add_argument('--out',default='hemisphere_check_01');p.add_argument('--atlas-template');p.add_argument('--build',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:
  h=classify(np.array([0,1002,2002,1005,2005,1121,2121,99]),{'V1':1002,'V2':1005})
  assert h.tolist()==[0,1,2,1,2,0,0,0]
  try:classify(np.array([1002.25]),{'V1':1002});raise AssertionError('fractional label accepted')
  except ValueError:pass
  print('PASS: exact LH/RH membership; unknown labels stay zero; fractional labels rejected. NIfTI checks await server.');return
 c=readj(a.config);out=Path(a.out).resolve();require(not out.exists(),'Output must be a new directory; use a new suffix')
 require('roi_size_sensitivity' in out.parts,'Run from roi_size_sensitivity or choose an output beneath that new directory');out.mkdir(parents=True)
 builder=str(Path(__file__).resolve().parents[1]/'02_registration_rois'/'build_allROI_nonoverlap_column.py')
 try:left=lookup(builder)
 except Exception as e:writej(out/'hemisphere_status.json',{'status':'BLOCKED','reason':str(e)});print(str(e));return
 templates=[a.atlas_template] if a.atlas_template else [PRIMARY,SECONDARY]
 require(not a.build or a.atlas_template,'For build, explicitly select --atlas-template after reviewing coverage; no automatic fallback selection')
 rows=[];summary=[];paths={};provenance=[]
 for template in templates:
  for s in SUBJECTS:
   path=expand(c,template,s)
   try:
    img,h,details,prov=assess(c,s,path,left);rows+=details
    unknown=sum(r['n_unknown'] for r in details);other=sum(r['n_other_parcel_atlas_label'] for r in details)
    ready=unknown==0 and other==0
    summary.append(dict(subject=s,atlas_path=str(path),status='READY_FOR_SOURCE_REVIEW' if ready else 'NEEDS_COVERAGE_OR_PARCEL_REVIEW',n_unknown=unknown,n_other_parcel_atlas_label=other,reason=''))
    if a.build and ready:
     import nibabel as nib
     dest=out/f'{s}_hemisphere_func.nii.gz';header=img.header.copy();header.set_data_dtype(np.uint8)
     nib.save(nib.Nifti1Image(h,img.affine,header),dest);paths[s]=str(dest);provenance.append(dict(subject=s,atlas_path=str(path),hemisphere_sha256=sha(dest),**prov))
   except Exception as e:summary.append(dict(subject=s,atlas_path=str(path),status='BLOCKED',reason=str(e)))
   csvwrite(out/'hemisphere_coverage.csv',rows);csvwrite(out/'hemisphere_sources.csv',summary)
 ready_config=a.build and len(paths)==15
 if ready_config:
  c['hemisphere_images']=paths;c['hemisphere_values']={'L':1,'R':2};c['hemisphere_provenance']=f'Exact atlas LH/RH label classification; source builder SHA256={BUILDER_SHA}; full selected-mask coverage, grid and parcel-label checks passed; no resampling/filling; source details in {out}/hemisphere_provenance.json. Historical atlas selection must be confirmed by the operator.'
  writej(out/'config_with_hemispheres.json',c)
 writej(out/'hemisphere_provenance.json',{'builder_path':str(builder),'builder_sha256':BUILDER_SHA,'source_role':'primary template is source-declared; secondary template was observed in server inventory, not verified as the final input','generated':provenance})
 writej(out/'hemisphere_status.json',{'status':'CONFIG_READY' if ready_config else 'CHECK_ONLY_OR_BLOCKED','config_written':bool(ready_config),'subjects_expected':15,'maps_written':len(paths),'scope':'Only 3D labels and BOLD headers read. Grid agreement and coverage do not independently prove historical registration lineage. Do not run original mask builder or drop unknown voxels.'})
 print('Reports:',out,'; config ready:',bool(ready_config))
if __name__=='__main__':main()
