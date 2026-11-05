"""Graph-consistency regressions, including a reported 9-tile SEM failure.

The reported shifts below reconstruct final shifts from the rounded coarse and
phase corrections in a user-supplied log. No microscope imagery is included.
"""
import unittest
import threading
from unittest.mock import patch

import numpy as np

from main import SEMStitcher, StitchingCancelled


REPORTED_EDGES = [
    (217, 0, 1, -686.12, 140.74),
    (545, 0, 5, -148.71, -571.19),
    (64, 1, 2, -821.89, -66.21),
    (100, 1, 4, -13.23, -714.98),
    (196, 2, 3, -15.10, -621.76),
    (121, 3, 4, 831.91, -17.01),
    (233, 3, 8, 242.47, -639.35),
    (253, 4, 5, 611.88, -8.39),
    (79, 4, 7, -12.90, -634.00),
    (811, 5, 6, -2.79, -621.70),
    (671, 6, 7, -555.82, -0.16),
    (827, 7, 8, -560.42, 1.15),
]


class GraphRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stitcher = SEMStitcher()
        self.messages = []
        self.stitcher.set_progress_callback(self.messages.append)

    def run_graph(self, count, edges):
        by_pair = {(i, j): (dx, dy, quality) for quality, i, j, dx, dy in edges}
        matches = [by_pair.get((i, j), (None, None, 0))
                   for i in range(count) for j in range(i + 1, count)]
        image = np.ones((884, 1024), np.uint8)
        with patch('main.cv2.imread', return_value=image), \
             patch.object(self.stitcher, '_try_match_pair', side_effect=matches), \
             patch.object(self.stitcher, '_compose_images', return_value=image) as compose:
            self.stitcher.stitch_sequence([f'tile_{i}.tif' for i in range(count)])
        return np.array(compose.call_args.args[1])

    def test_reported_twelve_edge_failure_recovers_with_one_exclusion(self):
        positions = self.run_graph(9, REPORTED_EDGES)
        residuals = self.stitcher._edge_residuals(REPORTED_EDGES, positions[:, 0], positions[:, 1])
        rejected = [(edge[1], edge[2]) for residual, edge in residuals if residual > 10]
        self.assertEqual(rejected, [(4, 5)])
        retained = [edge for residual, edge in residuals if residual <= 10]
        self.assertEqual(len(retained), 11)
        self.assertEqual(len(self.stitcher._connected_order(9, retained, 4)), 9)
        self.assertLess(max(residual for residual, edge in residuals if edge in retained), 9.1)
        self.assertTrue(any('tile_4.tif' in message and 'tile_5.tif' in message for message in self.messages))

    def test_coherent_reported_graph_keeps_all_remaining_edges(self):
        edges = [edge for edge in REPORTED_EDGES if edge[1:3] != (4, 5)]
        positions = self.run_graph(9, edges)
        residuals = self.stitcher._edge_residuals(edges, positions[:, 0], positions[:, 1])
        self.assertTrue(all(residual <= 10 for residual, edge in residuals))
        self.assertFalse(any('Exclude' in message for message in self.messages))

    def solve(self, count, edges, root=4):
        return self.stitcher._solve_consistent_graph(
            count, edges, root, 10.0, [f'tile_{i}.tif' for i in range(count)]
        )

    @staticmethod
    def regular_grid():
        positions = np.array([(col * 120.25, row * 100.375) for row in range(3) for col in range(3)])
        pairs = sorted(SEMStitcher._grid_neighbor_pairs(9, (3, 3), False))
        edges = [(100, i, j, *(positions[i] - positions[j])) for i, j in pairs]
        return positions, edges

    def test_high_quality_wrong_edge_is_found_even_when_its_own_residual_is_small(self):
        truth, edges = self.regular_grid()
        edges = [(10000, i, j, dx + 65, dy) if (i, j) == (4, 5) else (q, i, j, dx, dy)
                 for q, i, j, dx, dy in edges]
        initial = self.stitcher._solve_global_positions(9, edges, 4)
        self.assertLess(next(error for error, edge in self.stitcher._edge_residuals(edges, *initial[:2])
                             if edge[1:3] == (4, 5)), 10)
        retained, solution = self.solve(9, edges)
        self.assertEqual(len(retained), 11)
        self.assertFalse(any(edge[1:3] == (4, 5) for edge in retained))
        np.testing.assert_allclose(np.column_stack(solution[:2]), truth - truth[4], atol=1e-8)

    def test_recovery_preserves_a_bridge_and_its_image(self):
        edges = REPORTED_EDGES + [(30, 8, 9, -500.0, 3.0)]
        retained, solution = self.solve(10, edges)
        self.assertIn(edges[-1], retained)
        self.assertEqual(len(self.stitcher._connected_order(10, retained, 4)), 10)
        self.assertEqual(len(solution[0]), 10)

    def test_recovery_is_independent_of_edge_order_and_anchor(self):
        _, reference = self.solve(9, REPORTED_EDGES)
        reference_positions = np.column_stack(reference[:2])
        reference_positions -= reference_positions[0]
        for root in (0, 4, 8):
            with self.subTest(root=root):
                retained, solution = self.solve(9, list(reversed(REPORTED_EDGES)), root=root)
                actual = np.column_stack(solution[:2])
                np.testing.assert_allclose(actual - actual[0], reference_positions, atol=1e-8)
                self.assertEqual(len(retained), 11)

    def test_ambiguous_cycle_still_fails_instead_of_selecting_an_arbitrary_tree(self):
        edges = [(100, 0, 1, -100, 0), (100, 0, 2, -260, 0), (100, 1, 2, -100, 0)]
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous exclusions'):
            self.solve(3, edges, root=0)

    def test_multiple_conflicts_are_not_hidden_by_repeated_edge_removal(self):
        _, edges = self.regular_grid()
        edges = [(q, i, j, dx + (90 if (i, j) == (1, 4) else 0)
                   + (65 if (i, j) == (4, 5) else 0), dy) for q, i, j, dx, dy in edges]
        with self.assertRaisesRegex(RuntimeError, 'No single-edge exclusion restores consistency'):
            self.solve(9, edges)

    def test_recovery_can_be_cancelled(self):
        cancel = threading.Event()
        self.stitcher.set_cancel_callback(cancel.is_set)
        self.stitcher.set_progress_callback(lambda msg: cancel.set() if 'Checking' in msg else None)
        with self.assertRaises(StitchingCancelled):
            self.solve(9, REPORTED_EDGES)


if __name__ == '__main__':
    unittest.main()
