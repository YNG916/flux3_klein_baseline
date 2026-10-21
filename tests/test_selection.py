import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_data import fixture
from flux_mvwd.common import digest, file_hash, write_json
from flux_mvwd.data import Level1Data
from flux_mvwd.selection import make_selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = fixture(self.root)
        path = self.root / 'splits/train.jsonl'
        record = json.loads(path.read_text())
        records = [{**record, 'id': f'{scene}_{i}', 'scene': scene}
                   for scene in ('a', 'b', 'c') for i in range(5)]
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        meta = json.loads((self.root / 'dataset.json').read_text())
        meta['splits']['train']['count'] = len(records)
        meta['checksums']['splits/train.jsonl'] = file_hash(path)
        write_json(self.root / 'dataset.json', meta)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        data = Level1Data(self.config, 'train')
        with patch.object(data, 'target', side_effect=AssertionError('GT selection leak')):
            return make_selection(data, episodes_per_scene=2, seed=42)

    def test_balanced_reproducible_round_robin(self):
        first = self.manifest()
        self.assertEqual(first, self.manifest())
        self.assertEqual(len(first['episodes']), 6)
        for start in (0, 3):
            self.assertEqual({r['scene'] for r in first['episodes'][start:start+3]}, {'a', 'b', 'c'})
        path = self.root / 'selection.json'; write_json(path, first)
        data = Level1Data(self.config, 'train', selection=path)
        self.assertEqual([r['id'] for r in data.records], [r['id'] for r in first['episodes']])
        self.assertEqual(len(list(data.queries())), 36)
        self.assertEqual(data.provenance()['selection_sha256'], first['selection_sha256'])
        changed = make_selection(Level1Data(self.config, 'train'), episodes_per_scene=2, seed=43)
        self.assertNotEqual(first['episodes'], changed['episodes'])

    def test_tamper_wrong_split_protocol_and_overrides_rejected(self):
        original = self.manifest()
        for change, message in ((lambda m: m['episodes'][0].update(scene='wrong'), 'checksum'),
                                (lambda m: m.update(split='val'), 'checksum')):
            manifest = copy.deepcopy(original); change(manifest)
            path = self.root / 'selection.json'; write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, message):
                Level1Data(self.config, 'train', selection=path)
        path = self.root / 'selection.json'; write_json(path, original)
        with self.assertRaisesRegex(ValueError, 'different split'):
            Level1Data(self.config, 'val', selection=path)
        with self.assertRaisesRegex(ValueError, 'without scene'):
            Level1Data(self.config, 'train', selection=path, max_episodes=1)
        cfg = {**self.config, 'frame_sample_seed': 43}
        with self.assertRaisesRegex(ValueError, 'protocol'):
            Level1Data(cfg, 'train', selection=path)
        bad = copy.deepcopy(original)
        bad['episodes'][1] = bad['episodes'][0]
        bad['selection_sha256'] = digest({k: v for k, v in bad.items() if k != 'selection_sha256'})
        write_json(path, bad)
        with self.assertRaisesRegex(ValueError, 'unique'):
            Level1Data(self.config, 'train', selection=path)

    def test_cli_select_dispatches_metadata_only(self):
        from flux_mvwd.cli import main
        cfg = self.root / 'config.json'; write_json(cfg, self.config)
        output = self.root / 'cli_selection'
        argv = ['flux_mvwd', 'select', '--config', str(cfg), '--output', str(output),
                '--split', 'train', '--episodes-per-scene', '2']
        with patch('sys.argv', argv):
            main()
        self.assertEqual(json.loads((output / 'selection.json').read_text()), self.manifest())

    def test_no_silent_underfill_or_test_subset(self):
        with self.assertRaisesRegex(ValueError, 'fewer episodes'):
            make_selection(Level1Data(self.config, 'train'), episodes_per_scene=6)
        with self.assertRaisesRegex(ValueError, 'test'):
            make_selection(Level1Data(self.config, 'test'), episodes_per_scene=1)
