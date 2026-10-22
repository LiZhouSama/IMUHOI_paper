"""Combine native EgoIMU and augmented IMUHOI++ Best evaluations with receipts."""
from pathlib import Path
import csv,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu,write_json,sha,now
EGO=ROOT.parent/'EgoIMU'
OUT=ROOT/'outputs/analysis/egoimu_vs_imuhoipp_two_dataset_best_20261020'
RUNS={'OMOMO':ROOT/'outputs/IMUHOI++/original_recipe_k1_seed10_omomo_20261018',
      'BEHAVE':ROOT/'outputs/IMUHOI++/paper_recipe_aug_seed10_behave_20261019'}
KEYS=('obj_trans_err_fusion','hoi_err_fusion','mpjpe','root_trans_err','jitter','mpjre_angle')
PAPER={'OMOMO':dict(zip(KEYS[:5],(11.31,17.44,2.35,11.07,2.77))),
       'BEHAVE':dict(zip(KEYS[:5],(20.90,25.74,2.77,11.14,8.19)))}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def identities(rows,key):return {Path(r[key]).name for r in rows}
def clean(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    return v

def run():
    check_gpu(0)
    import torch,yaml
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]; receipts={}; deltas={}; mode_changes={}; provenance=[]; full_metrics={}
    for dataset,run_dir in RUNS.items():
        expected=1177 if dataset=='OMOMO' else 340
        native_paths={
            'offline':ROOT/'outputs/analysis/egoimu_native_eval_20261017/modules_best_test_disabled/results.json' if dataset=='OMOMO' else EGO/'outputs/offline_behave_seed10_20261020/results.json',
            'online':EGO/('outputs/online_omomo_seed10_20261019/results.json' if dataset=='OMOMO' else 'outputs/online_behave_seed10_20261020/results.json')}
        native={mode:read(path) for mode,path in native_paths.items()}
        assert all(p['status']=='complete' for p in native.values())
        key='sequences'
        native_ids={mode:identities(payload[key],'file_path' if dataset=='OMOMO' and mode=='offline' else 'sequence') for mode,payload in native.items()}
        assert all(len(ids)==expected for ids in native_ids.values())
        assert native_ids['offline']==native_ids['online']
        online=native['online'];assert (online['window'],online['lookahead'],online['chunk'],online['batch_size'])==(60,10,16,24)
        assert online['strict_tail'] and not online['eval_noise']
        for mode,payload in native.items():
            if not (dataset=='OMOMO' and mode=='offline'):
                assert len(payload['sequences'])==expected
                assert all(r['published_frames']==r['frames']-(10 if mode=='online' else 0) for r in payload['sequences'])
            checkpoints=payload.get('loaded_checkpoints',payload.get('checkpoints'))
            assert all(sha(Path(row['path']))==row['sha256'] for row in checkpoints.values())
        assert {k:r['sha256'] for k,r in native['offline'].get('loaded_checkpoints',native['offline'].get('checkpoints')).items()}=={k:r['sha256'] for k,r in native['online']['checkpoints'].items()}
        # Verify selected saved modules compose exactly the original joint-Best model.
        joint_path=EGO/'outputs/IMUHOI'/dataset.lower()/'stage_best_joint_training.pt'
        joint=torch.load(joint_path,map_location='cpu')['model_state_dict']
        tensor_count=0
        for name,row in native['online']['checkpoints'].items():
            module=torch.load(Path(row['path']),map_location='cpu')['module_state_dict']
            prefix=name+'_module.'
            selected={k[len(prefix):]:v for k,v in joint.items() if k.startswith(prefix)}
            assert set(selected)==set(module),(dataset,name)
            assert all(torch.equal(selected[k],v) for k,v in module.items())
            tensor_count+=len(module)
        original_sources=native['online']['source_sha256']
        for source,digest in original_sources.items():
            # OMOMO's CLI has since gained dataset selection; core math files stay immutable.
            if source!='eval_online_IMUHOI.py':assert sha(EGO/source)==digest,(dataset,source)
        if dataset=='BEHAVE':assert sha(EGO/'eval_online_IMUHOI.py')==original_sources['eval_online_IMUHOI.py']
        protocol=read(run_dir/'protocol.json')
        assert all(sha(Path(p))==digest for p,digest in protocol['frozen_sha256'].items()),dataset
        cfg=yaml.safe_load(Path(protocol['configs']['proposed_joint']).read_text())
        assert cfg['v5_init_noise_prob']==.5 and cfg['velocity_contact_arch']=='v5' and cfg['object_trans_arch']=='v2'
        assert cfg['object_trans_forearm_obj_rotation'] and cfg['object_trans_bimanual_closure']
        plus_values={}
        run_state=read(run_dir/'status.json')
        plus_checkpoint_receipts={}
        for mode in ('offline','online'):
            command=run_state['stages'][f'proposed_{mode}_best']['command']
            plus_checkpoint_receipts[mode]={flag:{'path':command[command.index(flag)+1],'sha256':sha(Path(command[command.index(flag)+1]))} for flag in ('--hp_ckpt','--vc_ckpt','--ot_ckpt')}
        assert plus_checkpoint_receipts['offline']==plus_checkpoint_receipts['online']
        for mode in ('offline','online'):
            path=run_dir/'proposed'/f'eval_{mode}_best.json'; value=read(path)
            cov_path=path.with_suffix('.coverage.json'); coverage=read(cov_path)
            assert coverage['status']=='passed'
            ids=identities(coverage['sequences'],'seq_path')
            assert len(coverage['sequences'])==len(ids)==expected and ids==native_ids[mode],(dataset,mode)
            assert value['hoi_metric_protocol']=='egoimu'
            if mode=='online':assert value['online_window']==60 and value['online_lookahead']==10
            plus_values[mode]=value['results'][dataset.lower()]
            for method,metrics,source in (('EgoIMU',native[mode]['results'],native_paths[mode]),('IMUHOI++',plus_values[mode],path)):
                assert all(math.isfinite(metrics[k]) for k in KEYS)
                row={'dataset':dataset,'method':method,'mode':mode,'checkpoint':'best','sequences':expected,**{k:metrics[k] for k in KEYS},'source':str(source)}
                full_metrics.setdefault(dataset,{}).setdefault(method,{})[mode]=clean(metrics)
                rows.append(row);provenance.append({'path':str(source),'sha256':sha(source)})
            for record in coverage['sequences']:
                assert all(math.isfinite(record[k]) for k in ('obj_cm','hoi_cm'))
        deltas[dataset]={mode:{k:plus_values[mode][k]-native[mode]['results'][k] for k in KEYS} for mode in ('offline','online')}
        mode_changes[dataset]={method:{k:values['online'][k]-values['offline'][k] for k in KEYS} for method,values in (
            ('EgoIMU',{m:native[m]['results'] for m in ('offline','online')}),('IMUHOI++',plus_values))}
        receipts[dataset]={'sequence_count':expected,'all_four_membership_sets_equal':True,'online_tail_contract':'T-10',
            'native_best_modules_equal_joint_best':True,'native_best_verified_tensors':tensor_count,
            'native_joint_best':{'path':str(joint_path),'sha256':sha(joint_path)},
            'native_checkpoints':native['online']['checkpoints'],'imuhoipp_best_checkpoints':plus_checkpoint_receipts['offline'],'imuhoipp_frozen_protocol_verified':True,
            'native_offline_rounds_to_paper':{k:round(native['offline']['results'][k],2)==v for k,v in PAPER[dataset].items()}}
    write_json(OUT/'comparison.json',clean({'created_at':now(),'rows':rows,'full_metrics':full_metrics,'imuhoipp_minus_egoimu':deltas,
        'online_minus_offline':mode_changes,'verification':receipts,'paper_table2':PAPER,'source_files':provenance,
        'scope':'Best only; augmented formal ++ runs, native original EgoIMU models/data. Data identities match, content and body/physics differ.'}))
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# EgoIMU 与 IMUHOI++：OMOMO / BEHAVE best 的 offline–online 对比','',f'生成时间：{now()}','',
        '所有行均为各自保存的 best 模型；IMUHOI++使用正式开启增广实验，不包含 final、速度替换或关闭增广诊断。',
        'Obj/HOI/MPJPE/Root单位cm，Ang单位度，Jitter按原代码 mm/frame² 数值口径；越低越好。','',
        '| 数据集 | 方法 | 模式 | 序列数 | Obj | HOI | MPJPE | Root | Jitter | Ang |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+' | '.join([r['dataset'],r['method'],r['mode'],str(r['sequences'])]+[f'{r[k]:.4f}' for k in KEYS])+' |')
    lines+=['','## 同模式方法差值','',
        '下表为 IMUHOI++ 减 EgoIMU；负数表示 ++ 较低。','',
        '| 数据集 | 模式 | ΔObj | ΔHOI | ΔMPJPE | ΔRoot | ΔJitter |','|---|---|---:|---:|---:|---:|---:|']
    for dataset,values in deltas.items():
        for mode,r in values.items():lines.append('| '+' | '.join([dataset,mode]+[f'{r[k]:+.4f}' for k in KEYS[:5]])+' |')
    lines+=['','## 口径与结果来源','',
        '- EgoIMU使用原版HP/VC/OT、原版预处理数据、SMPL-H male及原生evaluate_model；BEHAVE本次实测offline和online，OMOMO复用之前已核验的完整结果。',
        '- 两数据集各自四行成员完全一致：OMOMO为1177条；BEHAVE为原版min_obj_contact_frames=10保留的340个5–10秒片段，完整片段推理，不改用未裁剪长序列。',
        '- Online统一W60、lookahead10、chunk16、batch24；30Hz下固定10帧未来上下文，首块有warmup缓冲；只发布T−10帧，最后10帧不参与指标。Offline使用完整T帧，因此两模式不是同帧集合，差值不能纯粹解释为因果online代价。',
        '- 初始窗口用序列起点GT初始化；后续窗口由已发布预测初始化。不额外添加评估噪声；seed10。',
        '- EgoIMU三份best模块逐tensor拼接核验与各自stage_best_joint_training.pt完全相等，offline和online使用相同权重哈希；++两模式均加载同一shared HP best和joint VC/OT best。',
        '- HOI为EgoIMU手腕20/21接触相对误差，无手接触帧用root相对误差，按序列等权平均。',
        '- ++保留当前canonical预处理：OMOMO为SMPL-X neutral，BEHAVE为SMPL-H neutral/zero-betas；还保留当前物体平滑、接触GT及HP物理实现。这是同划分、同调度下各自流程的比较，不是同数据内容和同GT的纯架构消融，差值不能全部归因于++。',
        '- ++是seed10原配方预算对照：K1/W120、各阶段300轮，v5ar增广p0.5/σ0.2m/s、速度120+接触180课程及v2armbic专有损失；不将早期seed42/K4或延长速度训练的诊断冒充此处best。',
        '- best按各自训练保存的窗口验证规则选取，未按完整test指标倒选；只评估已有EgoIMU权重，本次未重训原版。','',
        '### 原版offline与论文Table2核对','',
        '| 数据集 | 项目 | Obj | HOI | MPJPE | Root | Jitter |','|---|---|---:|---:|---:|---:|---:|']
    for dataset in RUNS:
        for name,v in [('论文',PAPER[dataset]),('EgoIMU实测',next(r for r in rows if r['dataset']==dataset and r['method']=='EgoIMU' and r['mode']=='offline'))]:
            lines.append('| '+' | '.join([dataset,name]+[f'{v[k]:.4f}' for k in KEYS[:5]])+' |')
    lines+=['','### 可追溯结果文件','']
    for r in rows:
        path=str(Path(r['source'])).replace('/mnt/d/','D:/')
        lines.append(f'- {r["dataset"]} / {r["method"]} / {r["mode"]}: [{Path(path).name}]({path})')
    lines+=['','完整全指标和差值见 comparison.json；扁平表格见 comparison.csv。原BEHAVE online自检见EgoIMU/outputs/online_behave_seed10_20261020/self_test.json。']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(OUT/'comparison.md'),'rows':len(rows),'verification':receipts},ensure_ascii=False))
if __name__=='__main__':run()
