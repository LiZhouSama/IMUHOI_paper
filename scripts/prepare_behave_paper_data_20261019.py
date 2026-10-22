"""Rebuild current canonical BEHAVE crops using original dataset boundaries only."""
from pathlib import Path
import sys,json,collections,hashlib
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu,write_json,sha,now
OUT=ROOT/'outputs/IMUHOI++/paper_recipe_aug_seed10_behave_20261019'
EGO=ROOT.parent/'EgoIMU'

def run():
    check_gpu(0)
    import torch,numpy as np
    from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_rotation_6d
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data_audit.json').exists():
        print('Existing audited dataset preserved'); return
    current={p.stem:p for s in ('train','test') for p in (ROOT/'process/IMUHOI++cano/processed_split_data_BEHAVE'/s).glob('*.pt')}
    assert len(current)==len(set(current))
    required={p.stem.rsplit('_seg',1)[0] for split in ('train','test') for p in (EGO/'process/processed_seg_data_BEHAVE'/split).glob('*.pt')}
    missing=sorted(required-set(current))
    if missing:
        check_gpu(6000)
        from process.preprocess_BEHAVE import BodyModelLoader,process_behave_sequence
        device=torch.device('cuda:0'); loader=BodyModelLoader('datasets/smpl_models',device,num_betas=16)
        objects=ROOT/'datasets/BEHAVE/objects'; object_names=[p.name for p in objects.iterdir() if p.is_dir()]
        for name in missing:
            recovered=OUT/'canonical_sources'/f'{name}.pt'
            if not recovered.exists():
                result=process_behave_sequence(str(ROOT/'datasets/BEHAVE/sequences'/name),loader,str(objects),object_names,device,str(recovered.parent),None,256,256,skip_object_displacement_without_hand_contact=False)
                assert result is not None,name
            current[name]=recovered
        del loader
        torch.cuda.empty_cache()
    write_json(OUT/'canonical_recovery.json',{'missing_original_sources':missing,'recovered_with_current_preprocessor':True,'output_directory':str(OUT/'canonical_sources')})
    rows=[]; members={'train':[],'test':[]}; inventory={'train':[],'test':[]}
    for split in ('train','test'):
        groups=collections.defaultdict(list)
        for p in (EGO/'process/processed_seg_data_BEHAVE'/split).glob('*.pt'):
            name,index=p.stem.rsplit('_seg',1); groups[name].append((int(index),p))
        directory=OUT/'data'/split; directory.mkdir(parents=True,exist_ok=True)
        for name,segments in sorted(groups.items()):
            assert name in current, ('missing canonical source',name)
            source=current[name]; full=torch.load(source,map_location='cpu'); T=len(full['pose']); offset=0
            assert [i for i,p in sorted(segments)]==list(range(len(segments))),name
            for index,path in sorted(segments):
                old=torch.load(path,map_location='cpu'); L=len(old['rotation_local_full_gt_list']); end=offset+L
                assert end<=T,(name,end,T)
                error=float((torch.as_tensor(old['obj_trans'])-torch.as_tensor(full['obj_trans'][offset:end])).abs().max())
                pose6=matrix_to_rotation_6d(axis_angle_to_matrix(full['pose'][offset:end,:66].reshape(L,22,3))).reshape(L,132)
                pose_error=float((pose6-torch.as_tensor(old['rotation_local_full_gt_list'])).abs().max())
                assert pose_error<1e-4,('frame alignment mismatch',path.name,offset,pose_error)
                eligible=L>=121 and int(torch.as_tensor(old['obj_contact']).sum())>=10
                payload={k:(v[offset:end].clone() if torch.is_tensor(v) and v.ndim and len(v)==T else v[offset:end].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==T else v) for k,v in full.items()}
                payload.update(seq_name=path.stem,segment_start_frame=offset,segment_end_frame=end-1,source_sequence=name)
                if 'num_frames' in payload: payload['num_frames']=L
                destination=directory/path.name
                torch.save(payload,destination)
                stat=destination.stat(); inventory[split].append({'name':path.name,'sha256':sha(destination),'stat':[stat.st_size,stat.st_mtime_ns]})
                if eligible: members[split].append(path.name)
                rows.append({'split':split,'name':path.name,'source':str(source),'source_sha256':sha(source) if index==0 else None,'start':offset,'end_exclusive':end,'frames':L,'original_contact_frames':int(torch.as_tensor(old['obj_contact']).sum()),'current_contact_frames':int(torch.as_tensor(payload['obj_contact']).sum()),'original_eligible':eligible,'pose_rotation_6d_max_error':pose_error,'object_max_abs_error_m':error,'original_sha256':sha(path)})
                offset=end
            print(json.dumps({'split':split,'source':name,'saved':len(rows)}),flush=True)
    source_sets={s:{r['source'] for r in rows if r['split']==s} for s in ('train','test')}
    assert source_sets['train'].isdisjoint(source_sets['test'])
    write_json(OUT/'data/inventory.json',inventory)
    write_json(OUT/'data_audit.json',{'created_at':now(),'source':'Current canonical complete sequences; original data supply only split, frame boundaries and eligibility','counts':{s:{'all':len(inventory[s]),'eligible':len(members[s]),'sources':len(source_sets[s])} for s in ('train','test')},'members':members,'rows':rows,'source_overlap':False,'max_object_trajectory_error_m':max(r['object_max_abs_error_m'] for r in rows)})
    print('COMPLETE',json.dumps({s:(len(inventory[s]),len(members[s])) for s in members}))
if __name__=='__main__': run()



