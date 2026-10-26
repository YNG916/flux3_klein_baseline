"""Complete-split partitioning/merging must preserve seed, pixels and provenance."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from PIL import Image
from test_data import fixture
from flux_mvwd.common import digest, file_hash, read_json, write_json, sample_seed
from flux_mvwd.data import Level1Data
from flux_mvwd.preparation import identity, input_key
from flux_mvwd.runtime import source_hash

spec=importlib.util.spec_from_file_location('evaluation_launcher',Path(__file__).parents[1]/'slurm/evaluate_existing_allocation.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def setup_corpus(tmp_path):
    config=fixture(tmp_path/'dataset')
    root=Path(config['dataset_root']);meta=read_json(root/'dataset.json')
    split=root/'splits/test.jsonl';record=json.loads(split.read_text())
    split.write_text(''.join(json.dumps({**record,'id':f'ep{i}'})+'\n' for i in range(3)))
    meta['checksums']['splits/test.jsonl']=file_hash(split);meta['splits']['test']['count']=3
    write_json(root/'dataset.json',meta)
    data=Level1Data(config,'test');queries=list(data.queries())
    source=tmp_path/'captions_full';rows=[]
    for i,q in enumerate(queries):
        condition=data.conditions(q)
        name=f'prompts/{i:07d}.json'
        cap={'query':q,'prompt':'original exact instruction '+condition['prompt'],
             'input_sha256':input_key(condition),'ground_truth_loaded':False}
        write_json(source/name,cap)
        row={'query_index':i,'query':q,'path':name,'files':{name:file_hash(source/name)}}
        write_json(source/f'rows/{i:07d}.json',row);rows.append(row)
    write_json(source/'manifest.json',{**identity(config,data,'captions',queries),'complete':True,'samples':rows,'sample_count':len(rows)})
    return config,data,source


def test_partition_derivation_and_merge_keep_complete_protocol(tmp_path):
    config,data,source=setup_corpus(tmp_path)
    selections=launcher.partition(data,2)
    assert [len(s['episodes']) for s in selections]==[2,1]
    assert [e['id'] for s in selections for e in s['episodes']]==['ep0','ep1','ep2']
    pieces=[];impl=source_hash();seed=42
    for n,selection in enumerate(selections):
        sf=tmp_path/f'selection{n}.json';write_json(sf,selection)
        shard=Level1Data(config,'test',selection=sf)
        caps=tmp_path/f'caps{n}'
        launcher.derive_captions(config,source,caps,shard)
        launcher.derive_captions(config,source,caps,shard)  # restart is idempotent
        assert read_json(caps/'manifest.json')['sample_count']==len(list(shard.queries()))
        piece=tmp_path/f'piece{n}';rows=[]
        for i,q in enumerate(shard.queries()):
            robot=q['robots'][0];relative=f"{q['episode_id']}/robot_{robot:02d}/frame_{q['frame_index']:03d}.png"
            path=piece/relative;path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(96,64),(10+n,20,30)).save(path)
            cap=read_json(caps/f'prompts/{i:07d}.json')
            row={'episode_id':q['episode_id'],'frame_index':q['frame_index'],'robot_id':robot,
                'group_robots':q['robots'],'query_index':i,'path':relative,'sha256':file_hash(path),
                'seed':sample_seed(seed,data.meta['release_id'],q['episode_id'],q['frame_index'],q['robots'],config['mode']),
                'prompt_sha256':digest(cap['prompt']),'input_sha256':cap['input_sha256']}
            write_json(path.with_suffix('.generation.json'),{**row,'ground_truth_loaded':False})
            rows.append(row)
        (piece/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        write_json(piece/'metadata.json',{'schema':'flux3-mvwd-predictions-v1','complete':True,'ground_truth_loaded':False,
            'config':config,'base_seed':seed,'lora_sha256':'adapter','implementation_sha256':impl,
            'dataset':shard.provenance(),'versions':{'torch':'test'},'seed_policy':'independent-query-sha256',
            'view_count':len(rows),'caption_manifest_sha256':file_hash(caps/'manifest.json')})
        pieces.append(piece)
    target=tmp_path/'merged'
    launcher.merge_predictions(config,data,source,pieces,target,seed,'adapter',impl)
    launcher.merge_predictions(config,data,source,pieces,target,seed,'adapter',impl)
    meta=read_json(target/'metadata.json');rows=launcher.read_predictions(target)
    assert meta['complete'] and meta['view_count']==18 and not meta['ground_truth_loaded']
    assert meta['dataset']==data.provenance() and not meta['partition_merge']['prediction_pixels_modified']
    assert [r['query_index'] for r in rows]==list(range(18))
    assert all(file_hash(target/r['path'])==r['sha256'] for r in rows)
    with pytest.raises(ValueError,match='incomplete split coverage'):
        launcher.merge_predictions(config,data,source,pieces[:1],tmp_path/'missing',seed,'adapter',impl)
    with pytest.raises(ValueError,match='Duplicate'):
        launcher.merge_predictions(config,data,source,[*pieces,pieces[0]],tmp_path/'duplicate',seed,'adapter',impl)
    with pytest.raises(ValueError,match='incompatible'):
        launcher.merge_predictions(config,data,source,pieces,tmp_path/'wrong_model',seed,'different_adapter',impl)
    image=pieces[0]/launcher.read_predictions(pieces[0])[0]['path'];image.write_bytes(b'changed pixels')
    with pytest.raises(ValueError,match='pixels changed'):
        launcher.merge_predictions(config,data,source,pieces,tmp_path/'bad_pixels',seed,'adapter',impl)
