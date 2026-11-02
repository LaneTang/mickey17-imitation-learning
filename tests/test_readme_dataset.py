"""Evidence invariants for the new dataset visualizations (no hardware)."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('readme_figures', ROOT/'tools/build_readme_figures.py')
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


class DatasetEvidenceTests(unittest.TestCase):
    def test_groups_split_channels_and_final_frames(self):
        data, traces = figures.load_snapshot(ROOT/'evidence/2026-11-02-dataset')
        records = data['episodes']
        self.assertEqual(Counter((r['group'], r['split']) for r in records),
                         {('scene','train'):94, ('scene','validation'):16,
                          ('motion','train'):40, ('motion','validation'):8})
        self.assertEqual(sum(r['source_frames'] for r in records), 75570)
        split = json.loads((ROOT/'evidence/2026-11-01/split.json').read_text())
        self.assertEqual({r['episode'] for r in records if r['split']=='train'}, set(split['train_episodes']))
        for record in records:
            n = record['source_frames']
            self.assertEqual(record['sampled_frames'], len(set(range(0,n,3)) | {n-1}))
            rows = traces[record['episode']]
            self.assertTrue(all(len(row)==7 for row in rows))
            self.assertAlmostEqual(rows[0][0], record['first_time_s'], places=6)
            self.assertAlmostEqual(rows[-1][0], record['last_time_s'], places=6)
            self.assertTrue(all(a[0]<b[0] for a,b in zip(rows, rows[1:])))

    def test_current_checkpoint_matches_architecture(self):
        config = json.loads((ROOT/'evidence/2026-11-01/config.json').read_text())
        self.assertEqual((config['vision_backbone'],config['n_encoder_layers'],config['n_decoder_layers'],
                          config['chunk_size'], config['action_offset_frames']), ('resnet18',2,1,30,0))
        self.assertEqual(config['input_features']['observation.state']['shape'], [9])
        self.assertEqual(config['output_features']['action']['shape'], [9])
