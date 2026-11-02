"""Automatic timing uses observed motion, not task labels or target commands."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("completion", ROOT / "tools/analyze_completion_time.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def sequence(segments):
    rows = []
    for arm, grip, count in segments:
        for _ in range(count):
            state = [arm, 0, 0, 0, 0, grip]
            rows.append(dict(elapsed_s=len(rows) / 10, **dict(zip(a.COLUMNS, state))))
    return rows


def full_motion():
    return sequence([(0, 0, 10), (20, 0, 5), (20, 40, 5),
                     (20, 15, 5), (20, 40, 5), (0, 0, 10)])


class CompletionTimeTests(unittest.TestCase):
    def test_alpha_summary_uses_folder_membership_and_normal_candidates(self):
        spec = importlib.util.spec_from_file_location('completion_plot', ROOT / 'tools/plot_completion_times.py')
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
        trials = [dict(trial_id='one', memberships=['alpha/alpha-0.5'], alpha=.5,
                       candidate_completion_time_s=20, normal_ending_candidate=True,
                       rollout_status='time limit'),
                  dict(trial_id='error', memberships=['alpha/alpha-0.5'], alpha=.5,
                       candidate_completion_time_s=10, normal_ending_candidate=False,
                       rollout_status='error: stopped'),
                  dict(trial_id='extra', memberships=['base/2cm_right'], alpha=.5,
                       candidate_completion_time_s=1, normal_ending_candidate=True,
                       rollout_status='time limit')]
        report = dict(metric='observed_motion_place_and_return_candidate', trials=trials)
        summary = plot.alpha_results(report)[2]
        self.assertEqual((summary['n'], summary['timed_n'], summary['median_s'], summary['abnormal_n']), (2, 1, 20, 1))
        with self.assertRaises(ValueError):
            plot.alpha_results(dict(report, metric='placement_time'))

    def test_full_sequence_times_home_onset_not_confirmation(self):
        result = a.detect(full_motion(), [0]*6, a.DEFAULTS)
        self.assertEqual(result['candidate_completion_time_s'], 3.0)
        self.assertEqual(result['confirmed_at_s'], 3.5)

    def test_initial_hold_and_empty_return_do_not_count(self):
        for rows in [sequence([(0, 0, 100)]), sequence([(20, 0, 10), (0, 0, 10)])]:
            self.assertIsNone(a.detect(rows, [0]*6, a.DEFAULTS)['candidate_completion_time_s'])

    def test_release_without_return_does_not_count(self):
        self.assertIsNone(a.detect(full_motion()[:30], [0]*6, a.DEFAULTS)['candidate_completion_time_s'])

    def test_brief_home_crossing_and_open_gripper_do_not_count(self):
        for ending in [[(0, 0, 3), (20, 0, 10)], [(0, 40, 20)]]:
            rows = sequence([(20, 0, 5), (20, 40, 5), (20, 15, 5), (20, 40, 5)] + ending)
            self.assertIsNone(a.detect(rows, [0]*6, a.DEFAULTS)['candidate_completion_time_s'])

    def test_gap_breaks_home_dwell(self):
        rows = full_motion()[:34]
        rows[-1]['elapsed_s'] = 8.0
        self.assertIsNone(a.detect(rows, [0]*6, a.DEFAULTS)['candidate_completion_time_s'])

    def test_bad_time_and_nonfinite_data_rejected(self):
        for key, value in [('elapsed_s', -1), (a.COLUMNS[0], float('nan'))]:
            rows = full_motion()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                a.detect(rows, [0]*6, a.DEFAULTS)

    def test_deduplicates_and_ignores_even_unreadable_labels(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group in ['alpha/alpha-0.5', 'base/origin[baseline]']:
                run = root / group / 'act_offset0_overlap_example'
                run.mkdir(parents=True)
                (run/'manifest.json').write_text(json.dumps({'start_state':[0]*6, 'status':'error: stopped'}))
                (run/'quality_label.json').write_text('not valid JSON; must never be read')
                with (run/'cycles.csv').open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=['elapsed_s']+a.COLUMNS)
                    writer.writeheader()
                    writer.writerows(full_motion())
            trials = a.analyze(root, a.DEFAULTS)
            self.assertEqual(len(trials), 1)
            self.assertEqual(len(trials[0]['memberships']), 2)
            self.assertEqual(trials[0]['candidate_completion_time_s'], 3.0)
            self.assertFalse(trials[0]['normal_ending_candidate'])
            self.assertEqual(a.summarize(trials)['alpha/alpha-0.5']['normal_ending_candidates_n'], 0)
            (run/'cycles.csv').write_text('conflict')
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                a.analyze(root, a.DEFAULTS)
