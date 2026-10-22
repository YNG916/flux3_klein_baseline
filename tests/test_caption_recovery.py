"""Partial-caption migration preserves provenance and refuses changes to the task."""
import copy
from types import SimpleNamespace
import pytest
from test_data import fixture
from flux_mvwd.common import digest, file_hash, read_json, write_json
from flux_mvwd.data import Level1Data
from flux_mvwd.preparation import identity, input_key
from tools import caption_recovery as recovery


def test_partial_migration_checks_conditions_and_preserves_original(tmp_path, monkeypatch):
    config = fixture(tmp_path / 'data')
    config['prompt']['backend'] = 'vllm'
    new = copy.deepcopy(config)
    new['prompt']['overflow'] = {'policy':'concise_retry_then_exact_v1','retry_max_new_tokens':512}
    root = tmp_path / 'repo'
    source = root / 'artifacts/mvwd/experiments/old'
    source.mkdir(parents=True)
    write_json(source / 'config.json', config)
    sources = {}
    for name in ('flux_mvwd/data.py','flux_mvwd/camera_conditioning.py','flux_mvwd/common.py',
                 'flux_mvwd/preparation.py','flux_mvwd/runtime.py','UPSTREAM.lock.json','requirements.lock.txt'):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text('frozen contents')
        (source / 'source' / name).parent.mkdir(parents=True, exist_ok=True)
        (source / 'source' / name).write_text('frozen contents')
        sources[name] = file_hash(root / name)
    manifest = {'source_sha256': sources,'config_sha256':file_hash(source / 'config.json')}
    write_json(source / 'manifest.json', manifest)
    monkeypatch.setattr(recovery, 'ROOT', root)
    monkeypatch.setattr(recovery, 'SUPPORTED_IMPLEMENTATION', digest(sources))
    data = Level1Data(config,'train'); queries = list(data.queries())
    old_root = source / 'captions/train'
    old_identity = identity(config,data,'captions',queries)
    old_identity['implementation_sha256'] = digest(sources)
    write_json(old_root / 'identity.json',old_identity)
    condition = data.conditions(queries[0])
    caption = {'query':queries[0],'prompt':'description\n'+condition['prompt'],
        'ground_truth_loaded':False,'input_sha256':input_key(condition),
        'exact_constraints_sha256':digest(condition['prompt']),'text_tokens':100,'policy':config['prompt']}
    write_json(old_root / 'prompts/0000000.json', caption)
    write_json(old_root / 'rows/0000000.json', {'query_index':0,'query':queries[0],
        'path':'prompts/0000000.json','files':{'prompts/0000000.json':file_hash(old_root/'prompts/0000000.json')}})
    original = (old_root / 'prompts/0000000.json').read_bytes()
    output = source.parent / 'new/captions/import_step'
    args = SimpleNamespace(source_experiment='old',output=str(output),split='train')
    recovery.import_captions(new,args)
    result = read_json(output.parent / 'train/prompts/0000000.json')
    assert result['prompt'] == caption['prompt'] and result['imported_caption']['source_experiment'] == 'old'
    assert result['policy'] == config['prompt'] and not result['ground_truth_loaded']
    assert not (output.parent / 'train/manifest.json').exists()
    assert read_json(output / 'import_receipt.json')['imported_count'] == 1
    assert (old_root / 'prompts/0000000.json').read_bytes() == original
    changed = copy.deepcopy(new); changed['training']['learning_rate'] *= 2
    with pytest.raises(ValueError,match='Only addition'):
        recovery.validate_transition(config,changed,manifest)
    (root / 'flux_mvwd/data.py').write_text('different task')
    with pytest.raises(ValueError,match='Conditioning'):
        recovery.validate_transition(config,new,manifest)
