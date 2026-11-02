"""Exercise terminal input and labels without importing robot/policy modules."""

import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import pty
import select
import sys
import tempfile
import termios
import time
import tty
import unittest
from unittest.mock import patch


# Load the actual standalone definitions, excluding hardware imports and main.
SOURCE = Path(__file__).resolve().parents[1] / "deployment/rollout_act_overlap.py"
tree = ast.parse(SOURCE.read_text())
definitions = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))
               and node.name in {"CompletionKey", "save_operator_rating"}]
namespace = dict(os=os, select=select, sys=sys, termios=termios, time=time,
                 tty=tty, Path=Path, json=json, datetime=datetime, timezone=timezone)
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
CompletionKey = namespace["CompletionKey"]
save_operator_rating = namespace["save_operator_rating"]


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.master, slave = pty.openpty()
        self.stream = os.fdopen(slave, "r")
        self.original = termios.tcgetattr(self.stream.fileno())
        self.marker = CompletionKey(self.stream)
        self.marker.start(100.0, 40.0)

    def tearDown(self):
        self.marker.close()
        self.stream.close()
        os.close(self.master)

    def press(self, keys, now):
        os.write(self.master, keys)
        ready, _, _ = select.select([self.stream.fileno()], [], [], 1)
        self.assertTrue(ready)
        with patch.object(time, "perf_counter", return_value=now):
            self.marker.poll()

    def test_space_without_enter_and_first_press_wins(self):
        self.marker.poll()  # No input must return immediately.
        self.assertIsNone(self.marker.completion_time_s)
        self.press(b" ", 125.43)
        self.assertEqual(self.marker.completion_time_s, 25.43)
        self.press(b"   ", 130)
        self.assertEqual(self.marker.completion_time_s, 25.43)

    def test_other_keys_and_late_space_ignored(self):
        self.press(b"x", 110)
        self.assertIsNone(self.marker.completion_time_s)
        self.press(b" ", 140.1)
        self.assertIsNone(self.marker.completion_time_s)

    def test_terminal_restored_and_ctrl_c_enabled(self):
        active = termios.tcgetattr(self.stream.fileno())
        self.assertFalse(active[3] & termios.ICANON)
        self.assertTrue(active[3] & termios.ISIG)
        self.marker.close()
        restored = termios.tcgetattr(self.stream.fileno())
        # macOS may set the kernel-managed pending-input flag on restore.
        restored[3] &= ~getattr(termios, "PENDIN", 0)
        expected = list(self.original)
        expected[3] &= ~getattr(termios, "PENDIN", 0)
        self.assertEqual(restored, expected)
        self.marker.close()  # Cleanup is safe more than once.

    def test_nonterminal_is_noop(self):
        marker = CompletionKey(io.StringIO(" "))
        marker.start(100, 40)
        marker.poll()
        marker.close()
        self.assertIsNone(marker.completion_time_s)


class LabelTests(unittest.TestCase):
    def label(self, status, answers, completion=None):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sys, "stdin") as stream, \
                    patch("builtins.input", side_effect=answers), redirect_stdout(io.StringIO()):
                stream.isatty.return_value = True
                save_operator_rating(Path(folder), status, 100, completion)
            return json.loads((Path(folder) / "quality_label.json").read_text())

    def test_new_rank_and_only_one_completion_field(self):
        record = self.label("time limit", ["5", "0"], 25.43)
        self.assertEqual(record["quality_rank"], 5)
        self.assertEqual(record["completion_time_s"], 25.43)
        self.assertEqual(record["cube_in_the_circle"]["rating"], 0)
        self.assertEqual(record["format_version"], 3)
        for key in ("quality_score", "completed_phases", "fast_full_success", "score_out_of"):
            self.assertNotIn(key, record)
        self.assertEqual([k for k in record if k.startswith("completion_")], ["completion_time_s"])

    def test_no_phase_is_rank_one_and_unmarked_time_null(self):
        record = self.label("time limit", ["0", "", "1", ""])
        self.assertEqual(record["quality_rank"], 1)
        self.assertIsNone(record["completion_time_s"])
        self.assertIsNone(record["cube_in_the_circle"]["rating"])

    def test_error_preserves_completion_but_rank_null(self):
        record = self.label("error: controller violation", ["2"], 25.43)
        self.assertIsNone(record["quality_rank"])
        self.assertEqual(record["completion_time_s"], 25.43)
        self.assertEqual(record["cube_in_the_circle"]["rating"], 2)


if __name__ == "__main__":
    unittest.main()
