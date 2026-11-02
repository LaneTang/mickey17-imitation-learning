"""Outcome semantics and public evidence checks; no hardware imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis", ROOT / "tools/analyze_outcomes.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def trial(rank=0, status="time limit", time=None, missing=False):
    return {"quality_rank": rank, "status": status, "completion_time_s": time, "label_missing": missing}


class OutcomeSemantics(unittest.TestCase):
    def test_all_ranks(self):
        for rank in range(6):
            with self.subTest(rank=rank):
                self.assertEqual(a.outcome(trial(rank), 4), rank >= 4)
                self.assertEqual(a.outcome(trial(rank), 5), rank == 5)

    def test_abnormal_is_failure_even_with_rank_or_time(self):
        for status in ["error: ACT action queue ran dry; stopping", "operator Ctrl-C"]:
            self.assertIs(a.outcome(trial(5, status, 12.4), 4), False)
        self.assertIs(a.outcome(trial(None, "error: failed"), 4), False)

    def test_missing_and_unfinished_unknown(self):
        for t in [trial(None), trial(5, missing=True), trial(5, "started")]:
            self.assertIsNone(a.outcome(t, 4))

    def test_timing_uses_only_recorded_placements(self):
        s = a.group_summary([trial(4, time=12), trial(5, time=20), trial(5),
                             trial(0, time=3), trial(None, "error: failed", 1), trial(None, missing=True)])
        self.assertEqual((s['placement'], s['return'], s['unknown'], s['controller_errors']), (3, 2, 1, 1))
        self.assertEqual((s['timed_n'], s['placement_time_median_s']), (2, 16))
        self.assertEqual(s['n'], 6)

    def test_wilson_boundaries_and_known_value(self):
        self.assertEqual(a.wilson(0, 0), [None, None])
        self.assertAlmostEqual(a.wilson(8, 10)[0], .4901624715366417)
        self.assertAlmostEqual(a.wilson(0, 5)[1], .43448246478317476)
        self.assertAlmostEqual(a.wilson(5, 5)[1], 1.)


class EvidenceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "evidence/2026-11-01/outcomes.json").read_text())
        cls.summary, cls.audit = a.analyze(cls.data)

    def test_deduplication_and_membership(self):
        self.assertEqual(self.audit['unique_trials'], 107)
        self.assertEqual(self.audit['folder_memberships'], 127)
        self.assertEqual(self.audit['cohorts'], {'primary': 95, 'extra_base_right': 9, 'preliminary': 3})
        shared = [t for t in self.data['trials'] if len(t['memberships']) > 1]
        self.assertEqual(len(shared), 10)
        self.assertTrue(all(set(t['memberships']) == {'alpha/alpha-0.5', 'base/origin[baseline]',
                                                    'cube/deg0[baseline]'} for t in shared))
        for group in ['alpha/alpha-0.5', 'base/origin[baseline]', 'cube/deg0[baseline]']:
            self.assertEqual(self.summary[group]['placement'], 8)
            self.assertEqual(self.summary[group]['return'], 7)
        self.assertEqual(self.summary['base/2cm_right']['n'], 5)

    def test_missingness_and_errors(self):
        self.assertEqual(self.audit['primary_label_missing'], [])
        self.assertEqual(len(self.audit['all_label_missing']), 5)
        self.assertEqual(len(self.audit['placed_without_time']), 3)
        self.assertEqual(sum(self.audit['primary_controller_errors'].values()), 17)
        self.assertEqual(self.summary['cube/deg80']['timed_n'], 0)
        self.assertEqual(self.summary['cube/deg80']['placement'], 1)

    def test_deterministic_trace_selection(self):
        self.assertEqual(self.audit['selected_trajectory_ids'], [
            'act_offset0_overlap_20261030_221026_446645', 'act_offset0_overlap_20261030_222520_624170'])
        self.assertEqual([t['trial_id'] for t in self.data['trajectories']], self.audit['selected_trajectory_ids'])

    def test_reject_conflicting_duplicate_raw_folders(self):
        # Real shared-baseline import already checks hashes; simulate corruption.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            t = self.data['trials'][0]
            run = root / 'alpha/alpha-0.1' / t['trial_id']
            run.mkdir(parents=True)
            manifest = {'arm_filter_alpha': .1, 'status': 'started', 'checkpoint_sha256': a.MODEL_SHA,
                        'settings_sha256': 'x', 'controller_limits_sha256': 'y'}
            (run/'manifest.json').write_text(json.dumps(manifest))
            (run/'cycles.csv').write_text('cycle\n')
            (run/'chunks.csv').write_text('source_cycle,inference_ms\n')
            other = root / 'base/origin[baseline]' / t['trial_id']
            other.mkdir(parents=True)
            for name in ['manifest.json', 'cycles.csv', 'chunks.csv']:
                (other/name).write_bytes((run/name).read_bytes())
            (other/'extra.txt').write_text('conflicting member')
            # Use non-primary preliminary folder first to avoid final-cycle check.
            manifest['cycles'] = 0
            label = {'logged_cycles': 0, 'rollout_status': 'started'}
            for p in [run, other]:
                (p/'manifest.json').write_text(json.dumps(manifest))
                (p/'quality_label.json').write_text(json.dumps(label))
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                a.load_raw(root)

    def test_raw_import_preserves_missing_primary_label_as_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'alpha/alpha-0.1/act_offset0_overlap_missing_label'
            run.mkdir(parents=True)
            (run/'manifest.json').write_text(json.dumps({
                'arm_filter_alpha': .1, 'status': 'time limit', 'cycles': 0,
                'checkpoint_sha256': a.MODEL_SHA, 'settings_sha256': 'x',
                'controller_limits_sha256': 'y'}))
            (run/'cycles.csv').write_text('cycle\n')
            (run/'chunks.csv').write_text('source_cycle,inference_ms\n')
            result = a.load_raw(d)
            self.assertTrue(result['trials'][0]['label_missing'])
            self.assertIsNone(a.outcome(result['trials'][0], 4))


if __name__ == '__main__':
    unittest.main()
