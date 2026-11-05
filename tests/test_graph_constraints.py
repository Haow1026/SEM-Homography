import unittest

import numpy as np

from main import SEMStitcher


class GridConstraintTests(unittest.TestCase):
    def test_serpentine_grid_returns_only_cardinal_neighbours(self):
        pairs = SEMStitcher._grid_neighbor_pairs(
            n=9,
            grid_shape=(3, 3),
            serpentine=True,
        )
        self.assertEqual(
            pairs,
            {
                (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                (0, 5), (1, 4), (2, 3), (3, 8), (4, 7), (5, 6),
            },
        )

    def test_overlap_screen_rejects_corner_only_matches(self):
        image = np.empty((884, 1024), dtype=np.uint8)
        _, _, cardinal_ratio = SEMStitcher._overlap_geometry(image, image, 800, 0)
        _, _, diagonal_ratio = SEMStitcher._overlap_geometry(image, image, 800, 680)

        self.assertGreaterEqual(cardinal_ratio, 0.99)
        self.assertLess(diagonal_ratio, 0.30)

    def test_connected_order_excludes_disconnected_component(self):
        edges = [
            (10, 0, 1, 0.0, 0.0),
            (10, 1, 2, 0.0, 0.0),
        ]
        self.assertEqual(SEMStitcher._connected_order(4, edges, root=0), [0, 1, 2])


class GraphPipelineTests(unittest.TestCase):
    def setUp(self):
        self.stitcher = SEMStitcher()
        self.stitcher.set_progress_callback(lambda message: None)

    def run_matches(self, count, matches, **kwargs):
        from unittest.mock import patch
        image = np.ones((100, 100), np.uint8)
        with patch('main.cv2.imread', return_value=image), \
             patch.object(self.stitcher, '_try_match_pair', side_effect=matches):
            return self.stitcher.stitch_sequence([f'{i}.tif' for i in range(count)], **kwargs)

    def test_zero_area_overlap_is_rejected_even_with_one_full_axis(self):
        with self.assertRaisesRegex(RuntimeError, 'reliable edges'):
            self.run_matches(2, [(200, 0, 20)], min_axis_overlap_ratio=0)

    def test_enough_edges_do_not_imply_a_connected_graph(self):
        fail = (None, None, 0)
        with self.assertRaisesRegex(RuntimeError, 'Missing: 3.tif'):
            self.run_matches(4, [(-10, 0, 20), (-20, 0, 20), fail,
                                 (-10, 0, 20), fail, fail])

    def test_inconsistent_loop_cannot_silently_produce_a_panorama(self):
        with self.assertRaisesRegex(RuntimeError, 'Inconsistent image matches'):
            self.run_matches(3, [(-20, 0, 20), (-70, 0, 20), (-20, 0, 20)], residual_threshold=5)

    def test_consistent_fractional_constraints_recover_positions(self):
        positions = [(0, 0), (20.25, 0.5), (0.75, 30.375), (21, 30.875)]
        edges = [(20, i, j, positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
                 for i in range(4) for j in range(i + 1, 4)]
        x, y, *rest = self.stitcher._solve_global_positions(4, edges, 0)
        np.testing.assert_allclose(np.column_stack([x, y]), positions, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
