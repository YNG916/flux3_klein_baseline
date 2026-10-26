"""Restartable evaluation launcher; frozen training/inference modules stay unchanged."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux_mvwd.common import digest, file_hash, read_json, write_json, within, sample_seed
from flux_mvwd.data import Level1Data
from flux_mvwd.preparation import identity
from flux_mvwd.cache import initialize, existing_row, complete
from flux_mvwd.selection import PROTOCOL_KEYS
from flux_mvwd.runtime import source_hash
from types import SimpleNamespace


def partition(data, episodes):
    if episodes < 1:
        raise ValueError('Shard size must be positive')
    result = []
    for start in range(0, len(data.records), episodes):
        records = data.records[start:start + episodes]
        body = {'schema':'flux3-mvwd-selection-v1', 'release_id':data.meta['release_id'],
            'split':data.split, 'split_sha256':data.split_hash,
            'query_protocol':{k:data.config[k] for k in PROTOCOL_KEYS},
            'policy':{'name':'complete-split-contiguous-partition-v1','start':start,
                      'episodes_per_shard':episodes,'total_episodes':len(data.records)},
            'episodes':[{'id':r['id'],'scene':r['scene']} for r in records]}
        result.append({**body,'selection_sha256':digest(body)})
    flat = [r['id'] for m in result for r in m['episodes']]
    if flat != [r['id'] for r in data.records] or len(flat) != len(set(flat)):
        raise ValueError('Partition must cover the complete split exactly once')
    return result


def derive_captions(config, source, target, data):
    """Scope an existing GT-free caption cache to an exact partition; retain lineage."""
    source, target = Path(source), Path(target)
    full = Level1Data(config, data.split)
    full_queries = list(full.queries()); queries = list(data.queries())
    meta = read_json(source/'manifest.json')
    if not meta.get('complete') or any(meta.get(k)!=v for k,v in identity(config,full,'captions',full_queries).items()):
        raise ValueError('Source captions are not the complete matching split')
    source_sha = file_hash(source/'manifest.json')
    lookup = {digest(q):i for i,q in enumerate(full_queries)}
    info = identity(config,data,'captions',queries)
    acc = SimpleNamespace(is_main_process=True,wait_for_everyone=lambda:None)
    lock = initialize(target,info,acc)
    try:
        for i,q in enumerate(queries):
            j = lookup[digest(q)]; old = existing_row(source,j,q)
            if old is None: raise ValueError('Missing source caption')
            original = within(source,old['path']); cap = read_json(original)
            if cap['ground_truth_loaded'] is not False or cap['query']!=q:
                raise ValueError('Source caption is not GT-free or query differs')
            cap['derived_caption'] = {'source_manifest_sha256':source_sha,
                'source_query_index':j,'source_caption_sha256':file_hash(original)}
            previous = existing_row(target,i,q)
            if previous:
                if read_json(within(target,previous['path']))!=cap:
                    raise ValueError('Existing derived caption differs')
                continue
            name = f'prompts/{i:07d}.json'; write_json(target/name,cap)
            write_json(target/f'rows/{i:07d}.json',{'query_index':i,'query':q,'path':name,
                'files':{name:file_hash(target/name)}})
        complete(target,info,len(queries),acc)
        write_json(target/'derivation.json',{'source':str(source),'source_manifest_sha256':source_sha,
            'samples':len(queries),'ground_truth_loaded':False})
    finally:
        if lock: lock.close()


def prediction_key(row):
    return (row['episode_id'],row['frame_index'],row['robot_id'])


def read_predictions(path):
    return [json.loads(line) for line in (Path(path)/'predictions.jsonl').read_text().splitlines()]


def merge_predictions(config, data, captions, pieces, target, seed, lora_sha, implementation):
    """Reindex a disjoint complete partition; preserve generated pixels and all receipts."""
    target, captions = Path(target), Path(captions)
    queries = list(data.queries())
    expected = {(q['episode_id'],q['frame_index'],r):(i,q) for i,q in enumerate(queries) for r in q['robots']}
    full_caps = read_json(captions/'manifest.json')
    if not full_caps.get('complete') or any(full_caps.get(k)!=v for k,v in identity(config,data,'captions',queries).items()):
        raise ValueError('Merge requires complete matching full-split captions')
    source_rows = {}; receipts = []; first = None
    for piece in map(Path,pieces):
        m = read_json(piece/'metadata.json')
        if (not m.get('complete') or m.get('ground_truth_loaded') is not False
                or m['config']!=config or m['base_seed']!=seed or m['lora_sha256']!=lora_sha
                or m['implementation_sha256']!=implementation):
            raise ValueError('Incomplete or incompatible prediction partition')
        if any(m['dataset'][k]!=data.provenance()[k] for k in ('split','split_sha256','release_id','robot_assets')):
            raise ValueError('Partition dataset/assets differ')
        if first is None: first=m
        if m['versions']!=first['versions'] or m['seed_policy']!=first['seed_policy']:
            raise ValueError('Mixed generation environments or seed policies')
        rows = read_predictions(piece)
        if len(rows)!=m['view_count']: raise ValueError('Partition count differs')
        receipts.append({'directory':str(piece),'metadata_sha256':file_hash(piece/'metadata.json'),
            'predictions_sha256':file_hash(piece/'predictions.jsonl'),
            'caption_manifest_sha256':m['caption_manifest_sha256']})
        for row in rows:
            key=prediction_key(row)
            if key not in expected or key in source_rows: raise ValueError('Duplicate or unexpected prediction query')
            i,q=expected[key]
            if row['group_robots']!=q['robots'] or row['seed']!=sample_seed(seed,data.meta['release_id'],q['episode_id'],q['frame_index'],q['robots'],config['mode']):
                raise ValueError('Seed or camera grouping differs')
            caprow=existing_row(captions,i,q)
            if caprow is None: raise ValueError('Missing authoritative caption')
            cap=read_json(within(captions,caprow['path']))
            if row['prompt_sha256']!=digest(cap['prompt']) or row['input_sha256']!=cap['input_sha256']:
                raise ValueError('Generated prompt/input differs from authoritative caption')
            image=within(piece,row['path'])
            if file_hash(image)!=row['sha256']: raise ValueError('Prediction pixels changed')
            source_rows[key]=(piece,row,i)
    if set(source_rows)!=set(expected): raise ValueError('Cannot merge incomplete split coverage')
    meta={**first,'complete':False,'dataset':data.provenance(),'view_count':len(expected),
        'caption_manifest_sha256':file_hash(captions/'manifest.json'),
        'partition_merge':{'schema':'flux3-mvwd-complete-partition-merge-v1',
            'launcher_sha256':file_hash(Path(__file__)),'sources':receipts,'prediction_pixels_modified':False,
            'ground_truth_loaded':False}}
    target.mkdir(parents=True,exist_ok=True)
    if (target/'metadata.json').exists():
        old=read_json(target/'metadata.json')
        if {**old,'complete':False}!=meta: raise ValueError('Existing merged prediction provenance differs')
    else: write_json(target/'metadata.json',meta)
    merged=[]
    for key,(piece,row,i) in sorted(source_rows.items(),key=lambda kv:(kv[1][2],kv[0][2])):
        source=within(piece,row['path']); dest=within(target,row['path']);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if file_hash(dest)!=row['sha256']: raise ValueError('Existing merged image differs')
        else:
            try: os.link(source,dest)
            except OSError: shutil.copy2(source,dest)
        source_gen=source.with_suffix('.generation.json')
        if source_gen.exists():
            generation=read_json(source_gen)
            if generation['ground_truth_loaded'] is not False: raise ValueError('Generation sidecar used GT')
            generation.update(query_index=i,partition_origin={'directory':str(piece),
                'query_index':row['query_index'],'generation_sha256':file_hash(source_gen)})
            dest_gen=dest.with_suffix('.generation.json')
            if dest_gen.exists() and read_json(dest_gen)!=generation: raise ValueError('Merged generation sidecar differs')
            if not dest_gen.exists():write_json(dest_gen,generation)
        merged.append({**row,'query_index':i,'partition_origin':{'directory':str(piece),'query_index':row['query_index']}})
    text=''.join(json.dumps(r)+'\n' for r in merged)
    rowsfile=target/'predictions.jsonl'
    if rowsfile.exists() and rowsfile.read_text()!=text:raise ValueError('Existing merged manifest differs')
    if not rowsfile.exists():
        temporary=target/'predictions.jsonl.tmp';temporary.write_text(text);os.replace(temporary,rowsfile)
    write_json(target/'metadata.json',{**meta,'complete':True})
    return target


def run(args):
    exp=ROOT/'artifacts/mvwd/experiments'/args.experiment
    config=read_json(exp/'config.json');frozen=read_json(exp/'manifest.json')
    if file_hash(exp/'config.json')!=frozen['config_sha256'] or any(file_hash(ROOT/n)!=sha for n,sha in frozen['source_sha256'].items()):
        raise ValueError('Frozen config/source changed')
    budget=read_json(exp/'plans/full_training.json')['updates']
    finals=[f.parent for f in (exp/'training').glob('*/final/metadata.json')
        if read_json(f).get('complete') and read_json(f).get('step')==budget]
    if len(finals)!=1:raise ValueError('Expected one completed fixed-budget final checkpoint')
    checkpoint=finals[0]
    cm=read_json(checkpoint/'metadata.json');lora_sha=file_hash(checkpoint/'lora.safetensors')
    if cm['resume_contract']['implementation_sha256']!=source_hash() or cm['state_files']['lora.safetensors']!=lora_sha:
        raise ValueError('Final checkpoint source or adapter checksum differs')
    test=Level1Data(config,'test');selections=partition(test,args.episodes_per_shard)
    plan={'schema':'flux3-mvwd-formal-evaluation-plan-v1','implementation_sha256':source_hash(),
        'launcher_sha256':file_hash(Path(__file__)),'config_sha256':digest(config),
        'checkpoint':str(checkpoint.relative_to(exp)),'lora_sha256':lora_sha,
        'split':test.provenance(),'queries_per_seed':len(list(test.queries())),
        'seeds':[42,43,44],'sampler':config['inference'],'episodes_per_shard':args.episodes_per_shard,
        'shards_per_seed':len(selections),'selection_sha256':[s['selection_sha256'] for s in selections],
        'metrics':['psnr','lpips','fid','mv_gtwarp_dino'],'test_gt_used_for_selection':False}
    plan_path=exp/'plans/formal_evaluation.json'
    if plan_path.exists() and read_json(plan_path)!=plan:raise ValueError('Evaluation plan changed; keep its original launcher/config')
    if args.plan_only:
        print(json.dumps({k:plan[k] for k in ('queries_per_seed','seeds','shards_per_seed','sampler')},indent=2));return
    write_json(plan_path,plan)
    snapshot=exp/'source/evaluation'/f'formal_launcher_{plan["launcher_sha256"]}.py'
    if not snapshot.exists():snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,snapshot)
    lock=(exp/'reports/formal_evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def execute(kind,prefix,extra,processes=4,no_review=False):
        dest={'caption':'captions','infer':'predictions','eval':'evaluations'}[kind]
        for number in range(10000):
            name=f'{prefix}_a{number:03d}'
            receipt=exp/'steps'/f'{name}.json';output=exp/dest/name
            if receipt.exists():
                old=read_json(receipt)
                if old.get('status')=='completed' and old.get('exit_code')==0:
                    if kind=='infer' and not read_json(output/'metadata.json').get('complete'):raise ValueError('Successful receipt has incomplete predictions')
                    print('Reuse completed',name,flush=True);return output
                continue
            command=['./mvwd',kind,args.experiment,name]
            if kind in ('caption','infer'):command+=['--processes',str(processes)]
            if no_review:command+=['--no-review']
            command+=['--',*map(str,extra)]
            gpu_command=['srun','--jobid='+args.job_id,'--overlap','--nodes=1','--ntasks=1',
                '--cpus-per-task='+str(64 if processes==4 else 16),'--gres=gpu:'+str(processes),
                '--immediate=15','--kill-on-bad-exit=1','--chdir='+str(ROOT),*command]
            print('Execute',json.dumps(gpu_command),flush=True)
            subprocess.run(gpu_command,cwd=ROOT,check=True)
            return output
        raise RuntimeError('Too many previous attempts')
    def progress(stage,completed_views=0,**extra):
        write_json(exp/'reports/formal_evaluation_progress.json',{'stage':stage,'job_id':args.job_id,
            'completed_views':completed_views,'total_views':3*plan['queries_per_seed'],
            'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),**extra})
    captions=exp/'captions/test'
    progress('caption_test',ground_truth_loaded=False)
    execute('caption','formal_caption_test',['--split','test','--cache-root',captions])
    progress('derive_partition_captions',ground_truth_loaded=False)
    for i,selection in enumerate(selections):
        s=exp/f'selections/formal_test_{i:03d}/selection.json'
        if s.exists() and read_json(s)!=selection:raise ValueError('Partition selection changed')
        if not s.exists():write_json(s,selection)
        data=Level1Data(config,'test',selection=s)
        derive_captions(config,captions,exp/f'captions/formal_test_{i:03d}',data)
    outputs=[]; completed_views=0
    for seed in plan['seeds']:
        pieces=[]
        for i in range(len(selections)):
            selection=exp/f'selections/formal_test_{i:03d}/selection.json'
            progress('generation',completed_views,seed=seed,partition=i,ground_truth_loaded=False)
            pieces.append(execute('infer',f'formal_s{seed}_part{i:03d}',[
                '--split','test','--selection',selection,'--captions',exp/f'captions/formal_test_{i:03d}',
                '--lora',checkpoint,'--seed',seed],no_review=True))
            completed_views+=len(selections[i]['episodes'])*config['frames_per_episode']*len(config['robots'])
            progress('generation',completed_views,seed=seed,partition=i,ground_truth_loaded=False)
        merged=merge_predictions(config,test,captions,pieces,exp/f'predictions/formal_test_seed{seed}',seed,lora_sha,source_hash())
        outputs.append(merged)
        write_json(exp/'reports/formal_evaluation_progress.json',{'stage':'generation','complete_seeds':list(plan['seeds'][:len(outputs)]),
            'completed_views':len(outputs)*plan['queries_per_seed'],'total_views':3*plan['queries_per_seed'],
            'ground_truth_loaded':False,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    progress('metrics',completed_views,predictions_complete=True,gt_used_only_for_evaluation=True)
    result=execute('eval','formal_test_four_metrics',['--predictions',*outputs,
        '--metrics-repo',ROOT.parent/'top2pano','--weights-dir',ROOT.parent/'top2pano/models/metrics','--device','cuda'],processes=1)
    metrics=read_json(result/'metrics.json')
    if not metrics.get('benchmark_ready'):raise ValueError('Formal metrics failed coverage/backend checks')
    write_json(exp/'reports/formal_evaluation_progress.json',{'stage':'complete','metrics_directory':str(result),
        'completed_views':3*plan['queries_per_seed'],'total_views':3*plan['queries_per_seed'],
        'benchmark_ready':True,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    print('Formal evaluation complete:',result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id');parser.add_argument('experiment')
    parser.add_argument('--episodes-per-shard',type=int,default=100)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    if not args.job_id.isdigit() or Path(args.experiment).name!=args.experiment or args.experiment in ('.','..'):
        parser.error('Invalid allocation/experiment')
    run(args)

if __name__=='__main__':main()
