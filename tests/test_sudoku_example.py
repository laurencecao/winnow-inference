"""Offline checks for examples/sudoku's batched scoring strategy.

These tests never touch Winnow: the solver only requires a duck-typed scorer.
They pin the behaviour that matters for the batch strategy -- one batched
request instead of one request per branch point, graceful degradation when the
model fails, and the complete-search fallback that keeps solutions safe.
"""

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "sudoku"))

from sudoku_solver import SolveStats, SudokuSolver, validate_solution  # noqa: E402

MEDIUM = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
HARD = "400000805030000000000700000020000060000080400000010000000603070500200000104000000"
EVIL = "100007090030020008009600500005300900010080002600004000300000010040000007007000300"


def board(text):
    return [0 if ch in ".0" else int(ch) for ch in text]


class RecordingBatchScorer:
    """Implements the optional batch API and records every request."""

    def __init__(self, mode="uniform"):
        self.mode = mode
        self.calls = []
        self.stats = type("Stats", (), {"calls": 0, "questions": 0})()

    def score_candidates(self, board_, candidates):
        self.calls.append({cell: tuple(values) for cell, values in candidates.items()})
        self.stats.calls += 1
        self.stats.questions += len(candidates)
        scores = {}
        for cell, values in candidates.items():
            width = len(values)
            for value in values:
                if self.mode == "uniform":
                    scores[(cell, value)] = 1.0 / width
                elif self.mode == "adversarial":
                    # Every probability on the wrong end of the ordering: pruning
                    # must be able to lose the solution and the fallback must find it.
                    scores[(cell, value)] = 1.0 if value == max(values) else 0.0
                else:
                    scores[(cell, value)] = 0.0
        return scores


class OneCellOnlyScorer:
    """Only the legacy per-cell API; the solver must fall back to it."""

    def __init__(self):
        self.calls = 0
        self.stats = type("Stats", (), {"calls": 0, "questions": 0})()

    def score_one_cell(self, board_, cell, values):
        self.calls += 1
        self.stats.calls += 1
        self.stats.questions += 1
        return {value: 1.0 / len(values) for value in values}


class FailingScorer:
    def __init__(self):
        self.calls = 0
        self.stats = type("Stats", (), {"calls": 0, "questions": 0})()

    def score_candidates(self, board_, candidates):
        self.calls += 1
        raise RuntimeError("model is offline")


class OverlapDetectingScorer:
    """Sleeps inside the call so concurrent batches would overlap if not serialized."""

    def __init__(self):
        self.calls = 0
        self.overlapped = False
        self._active = 0
        self._lock = threading.Lock()
        self.stats = type("Stats", (), {"calls": 0, "questions": 0})()

    def score_candidates(self, board_, candidates):
        with self._lock:
            self._active += 1
            if self._active > 1:
                self.overlapped = True
            self.calls += 1
            self.stats.calls += 1
            self.stats.questions += len(candidates)
        time.sleep(0.01)
        with self._lock:
            self._active -= 1
        return {
            (cell, value): 1.0 / len(values)
            for cell, values in candidates.items()
            for value in values
        }


class BatchedStrategy(unittest.TestCase):
    def test_propagation_solves_medium_without_model(self):
        scorer = RecordingBatchScorer()
        solution = SudokuSolver(scorer).solve(board(MEDIUM))
        self.assertTrue(validate_solution(solution))
        self.assertEqual(scorer.calls, [])
        self.assertEqual(scorer.stats.questions, 0)

    def test_hard_needs_exactly_one_batched_request(self):
        scorer = RecordingBatchScorer()
        solver = SudokuSolver(scorer)
        solution = solver.solve(board(HARD))
        self.assertTrue(validate_solution(solution))
        self.assertEqual(len(scorer.calls), 1)
        self.assertGreater(scorer.stats.questions, 10, "batch should cover many cells")
        self.assertGreater(solver.stats.branches, 1)
        self.assertEqual(solver.stats.scorer_batches, 1)

    def test_batch_scores_are_reused_for_the_whole_search(self):
        scorer = RecordingBatchScorer()
        solver = SudokuSolver(scorer)
        solver.solve(board(EVIL))
        self.assertEqual(len(scorer.calls), 1, "one batch, reused by every branch point")
        self.assertEqual(solver.stats.scorer_errors, 0)

    def test_batch_request_only_covers_nearby_candidate_widths(self):
        scorer = RecordingBatchScorer()
        SudokuSolver(scorer, batch_slack=1).solve(board(HARD))
        widths = {len(values) for request in scorer.calls for values in request.values()}
        self.assertLessEqual(max(widths), 3, "slack=1 keeps the first batch narrow")

    def test_solver_falls_back_to_per_cell_scorer(self):
        scorer = OneCellOnlyScorer()
        solver = SudokuSolver(scorer)
        solution = solver.solve(board(HARD))
        self.assertTrue(validate_solution(solution))
        self.assertGreaterEqual(scorer.calls, 1)
        # The solver still batches its bookkeeping; only the transport is per cell.
        self.assertEqual(solver.stats.scorer_batches, 1)
        self.assertEqual(scorer.calls, scorer.stats.questions)

    def test_model_failure_degrades_to_uniform_once(self):
        scorer = FailingScorer()
        solver = SudokuSolver(scorer, prune=False)
        solution = solver.solve(board(HARD))
        self.assertTrue(validate_solution(solution))
        self.assertEqual(scorer.calls, 1, "a failed batch must not be retried cell by cell")
        self.assertGreaterEqual(solver.stats.scorer_errors, 1)

    def test_adversarial_scores_still_return_a_solution_via_fallback(self):
        scorer = RecordingBatchScorer(mode="adversarial")
        solver = SudokuSolver(scorer)
        solution = solver.solve(board(HARD))
        self.assertTrue(validate_solution(solution))
        self.assertTrue(solver.stats.prune_fallback)
        self.assertEqual(len(scorer.calls), 1, "fallback pass reuses the cached scores")

    def test_node_strategy_keeps_per_branch_scoring(self):
        scorer = OneCellOnlyScorer()
        solver = SudokuSolver(scorer, score_strategy="node", threshold=None, prune=False)
        solution = solver.solve(board(HARD))
        self.assertTrue(validate_solution(solution))
        self.assertEqual(scorer.calls, solver.stats.branches)
        self.assertEqual(solver.stats.scorer_batches, 0)

    def test_invalid_strategy_and_slack_are_rejected(self):
        with self.assertRaises(ValueError):
            SudokuSolver(score_strategy="turbo")
        with self.assertRaises(ValueError):
            SudokuSolver(batch_slack=-1)

    def test_parallel_branches_do_not_issue_overlapping_batches(self):
        scorer = OverlapDetectingScorer()
        solver = SudokuSolver(scorer, parallel=True, max_workers=4)
        solution = solver.solve(board(HARD))
        self.assertTrue(validate_solution(solution))
        self.assertGreaterEqual(scorer.calls, 1)
        self.assertFalse(scorer.overlapped, "batch calls must be serialized by the solver lock")

    def test_stats_defaults_stay_serializable(self):
        stats = SolveStats()
        self.assertIn("scorer_batches=0", stats.summary())


class SolverGeometry(unittest.TestCase):
    def test_unsolvable_board_returns_none(self):
        # Two 5s in the first row: is_consistent() is False, solve() returns None.
        puzzle = list(board(HARD))
        puzzle[0], puzzle[1] = 5, 5
        self.assertIsNone(SudokuSolver(None).solve(puzzle))

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            SudokuSolver(None).solve([0] * 80)


if __name__ == "__main__":
    unittest.main()
