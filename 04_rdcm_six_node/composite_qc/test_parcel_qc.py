"""Synthetic checks only: python3 test_parcel_qc.py. No server data needed."""
import copy
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from scipy.io import savemat
from parcel_qc import cell_metrics, run, sha, subject_paths


def main():
    signal=np.sin(np.arange(218)*.13)
    _,rows,_=cell_metrics(np.tile(signal,(3,1)),['a','b','c'])
    assert all(abs(r['parcel_to_rest_r']-1)<1e-12 and r['loo_centered_rmse_over_full_sd']<1e-12 for r in rows)
    x=np.array([signal,2*signal,-signal])
    _,rows,_=cell_metrics(x,['a','b','c'])
    assert rows[2]['parcel_to_rest_r']<-.999999
    assert rows[1]['status']=='UNDEFINED_CORRELATION'
    assert abs(sum(r['variance_contribution_signed'] for r in rows)-1)<1e-12
    full=x.mean(axis=0)
    for i,row in enumerate(rows):
        assert np.isclose(row['count_adjusted_change'],(x[i]-full).std()/full.std())
    config=json.loads(Path(__file__).with_name('config.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        cfg=copy.deepcopy(config);cfg['project_root']=tmp+'/inputs';cfg['subjects']=['sub-01','sub-02']
        names=[p for ps in cfg['members'].values() for p in ps]
        rng=np.random.default_rng(12)
        for subject in cfg['subjects']:
            folder,refs=subject_paths(cfg,subject);folder.mkdir(parents=True)
            x=(100+signal+rng.normal(0,.3,(len(names),3,218))).astype(np.float32)
            np.save(folder/'ts_matrix.npy',x)
            np.save(folder/'roi_names.npy',np.array(names,dtype=object))
            np.save(folder/'n_voxels_clean.npy',np.full((len(names),3),30,dtype=np.int32))
            (folder/'pial_exclusion_metadata.json').write_text(json.dumps(dict(subject=subject,n_rois=len(names),n_layers=3,n_timepoints=218)))
            composite={s:np.stack([x[[names.index(p) for p in ps],d,:].mean(axis=0) for d in range(3)],axis=1).astype(float) for s,ps in cfg['members'].items()}
            for model,path in refs.items():
                path.parent.mkdir(parents=True,exist_ok=True)
                savemat(path,dict(ts=np.column_stack([composite[s] for s in cfg['reference_stages'][model]]),
                    labels=np.array(cfg['reference_labels'][model],dtype=object),Subject=subject,TR=2.2,TE=.026,
                    n_nodes=len(cfg['reference_labels'][model]),stage_members_json=json.dumps(cfg['members']),validation_branch='pial5_current'))
        originals={p:sha(p) for p in Path(cfg['project_root']).rglob('*') if p.is_file()}
        out=Path(tmp)/'reports'
        result=run(cfg,out)
        assert result['execution_status']=='COMPLETE' and result['n_subjects_complete']==2
        assert (result['n_cells'],result['n_pairwise_rows'],result['n_loo_rows'])==(18,612,138)
        assert sum(1 for _ in (out/'node_reconstruction.csv').open())==31
        assert all(sha(p)==h for p,h in originals.items())
        with zipfile.ZipFile(out/'parcel_qc_return.zip') as z:
            assert len(z.namelist())==10
            assert all(Path(n).suffix in ('.json','.csv','.md') and n.startswith('reports/') for n in z.namelist())
        try:run(cfg,out)
        except FileExistsError:pass
        else:raise AssertionError('Existing reports overwritten')
        # A real parity discrepancy blocks that participant, with no selective parcel deletion.
        folder,_=subject_paths(cfg,'sub-01');x=np.load(folder/'ts_matrix.npy');x[0,0,0]+=1
        np.save(folder/'ts_matrix.npy',x)
        result=run(cfg,Path(tmp)/'mismatch')
        assert result['execution_status']=='INCOMPLETE' and result['n_subjects_complete']==1
        assert result['n_cells']==9 and 'Reconstruction parity failed' in result['subjects'][0]['reason']
        # Canonical duplicate names also stop QC rather than silently selecting a column.
        folder,_=subject_paths(cfg,'sub-02');bad_names=names.copy();bad_names[1]=bad_names[0]
        np.save(folder/'roi_names.npy',np.array(bad_names,dtype=object))
        result=run(cfg,Path(tmp)/'duplicates')
        assert result['n_subjects_complete']==0 and result['n_loo_rows']==0
        assert 'canonical duplicates' in result['subjects'][1]['reason']
        (folder/'ts_matrix.npy').unlink()
        result=run(cfg,Path(tmp)/'missing')
        assert 'FileNotFoundError' in result['subjects'][1]['reason'] and result['n_loo_rows']==0
    print('PASS: synthetic metric identities, MATLAB/NumPy schema, parity gate, missing/duplicate protection, report ZIP, unchanged inputs, no overwrites.')


if __name__=='__main__':main()
