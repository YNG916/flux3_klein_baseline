"""Explicit migration of successful v2 captions after the overflow-policy repair."""
from pathlib import Path
import copy
from types import SimpleNamespace
from flux_mvwd.common import digest, file_hash, read_json, write_json, within
from flux_mvwd.runtime import ROOT
from flux_mvwd.data import Level1Data
from flux_mvwd.preparation import identity, input_key
from flux_mvwd.cache import initialize, existing_row
from flux_mvwd.caption_worker import overflow_settings

# This importer supports the audited 497044e implementation only. It is not a
# general way to bypass immutable source/config contracts.
SUPPORTED_IMPLEMENTATION = '4e1a2dd8fcf840a39698051dc14d69dc71a0bb44108c80733e6caef4b107a40b'


def validate_transition(old_config, new_config, source_manifest):
    expected = copy.deepcopy(new_config)
    if overflow_settings(expected) is None:
        raise ValueError('Destination requires the explicit overflow repair policy')
    expected['prompt'].pop('overflow')
    if old_config != expected or old_config['prompt'].get('backend') != 'vllm':
        raise ValueError('Only addition of the overflow policy may differ')
    if digest(source_manifest['source_sha256']) != SUPPORTED_IMPLEMENTATION:
        raise ValueError('Unsupported source implementation for caption migration')
    # These files define image content, exact task/geometry, token budget and model
    # identity. The initial vLLM request/sampling policy is unchanged by the repair.
    for name in ('flux_mvwd/data.py', 'flux_mvwd/camera_conditioning.py',
                 'flux_mvwd/common.py', 'flux_mvwd/preparation.py',
                 'flux_mvwd/runtime.py', 'UPSTREAM.lock.json', 'requirements.lock.txt'):
        if file_hash(ROOT / name) != source_manifest['source_sha256'][name]:
            raise ValueError('Conditioning or environment changed: ' + name)


def import_captions(config, args):
    name = args.source_experiment
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Source experiment must be a simple name')
    output = Path(args.output)
    source = ROOT / 'artifacts/mvwd/experiments' / name
    if source.resolve() == output.parent.parent.resolve():
        raise ValueError('Import requires a new destination experiment')
    manifest = read_json(source / 'manifest.json')
    if file_hash(source / 'config.json') != manifest['config_sha256']:
        raise ValueError('Source config checksum mismatch')
    for relative, expected in manifest['source_sha256'].items():
        if file_hash(source / 'source' / relative) != expected:
            raise ValueError('Source snapshot checksum mismatch: ' + relative)
    validate_transition(read_json(source / 'config.json'), config, manifest)
    data = Level1Data(config, args.split)
    queries = list(data.queries())
    old_root = source / 'captions' / args.split
    old_identity = read_json(old_root / 'identity.json')
    old_config = read_json(source / 'config.json')
    expected = identity(old_config, data, 'captions', queries)
    expected['implementation_sha256'] = SUPPORTED_IMPLEMENTATION
    if old_identity != expected:
        raise ValueError('Source query support, data/assets, config or environment differs')
    new_root = output.parent / args.split
    if (new_root / 'manifest.json').exists():
        raise ValueError('Destination cache already complete; do not import over it')
    current_identity = identity(config, data, 'captions', queries)
    cpu = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None)
    lock = initialize(new_root, current_identity, cpu)
    imported = []
    try:
        for row_path in sorted((old_root / 'rows').glob('*.json')):
            row = read_json(row_path)
            i = row['query_index']
            if type(i) is not int or not 0 <= i < len(queries):
                raise ValueError('Invalid source query index')
            old = existing_row(old_root, i, queries[i])
            if old is None or row != old:
                raise ValueError('Invalid source row identity')
            cap_path = within(old_root, old['path'])
            cap = read_json(cap_path)
            condition = data.conditions(queries[i])
            if (cap['query'] != queries[i] or cap['ground_truth_loaded'] is not False
                    or cap['input_sha256'] != input_key(condition)
                    or cap['exact_constraints_sha256'] != digest(condition['prompt'])
                    or not cap['prompt'].endswith(condition['prompt'])
                    or not 0 < cap['text_tokens'] <= 4096
                    or cap['policy'] != old_config['prompt']):
                raise ValueError('Source caption conditions differ')
            lineage = {'source_experiment': name, 'source_query_index': i,
                'source_implementation_sha256': SUPPORTED_IMPLEMENTATION,
                'source_config_sha256': digest(old_config),
                'source_caption_sha256': file_hash(cap_path),
                'source_row_sha256': file_hash(row_path),
                'source_identity_sha256': file_hash(old_root / 'identity.json')}
            new_cap = {**cap, 'caption_status': 'imported_v2_ok', 'upsampler_caption_used': True,
                       'imported_caption': lineage}
            relative = f'prompts/{i:07d}.json'
            new_row = {'query_index': i, 'query': queries[i], 'path': relative,
                       'files': {}, 'imported_caption': lineage}
            previous = existing_row(new_root, i, queries[i])
            if previous:
                if read_json(new_root / previous['path']) != new_cap:
                    raise ValueError('Destination caption already differs; refusing overwrite')
            else:
                write_json(new_root / relative, new_cap)
                new_row['files'][relative] = file_hash(new_root / relative)
                write_json(new_root / 'rows' / f'{i:07d}.json', new_row)
            imported.append(i)
            if len(imported) % 256 == 0:
                print(f'Imported verified captions {len(imported)}', flush=True)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'import_receipt.json', {'schema': 'flux3-mvwd-caption-recovery-v1',
            'source_experiment': name, 'source_manifest_sha256': file_hash(source / 'manifest.json'),
            'source_identity_sha256': file_hash(old_root / 'identity.json'),
            'destination_cache': str(new_root), 'destination_identity': current_identity,
            'imported_queries': imported, 'imported_count': len(imported),
            'cache_complete': False, 'ground_truth_loaded': False,
            'reason': 'Reuse validated successful initial outputs; explicit overflow recovery affects failures only.'})
        print(f'Imported {len(imported)} verified captions into {new_root}; full generation remains incomplete', flush=True)
    finally:
        if lock: lock.close()

