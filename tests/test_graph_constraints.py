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


if __name__ == "__main__":
    unittest.main()
