import unittest

import numpy as np
from draw_picture import get_cutoff_points, get_corners_of_cut_texture, add_dummy_point, get_model_comparator


class TestCutoffPointCalculation(unittest.TestCase):
    def setUp(self):
        # Create a flat square with side lengths 20 and center in (0, 0, 0)
        self.flat_square = np.array([
            [-10, -10, 0],  # Corner C1
            [-10, 10, 0],   # Corner C2
            [10, 10, 0],    # Corner C3
            [10, -10, 0]    # Corner C4
        ])

        # Define the orientation to be looking straight at (0, 0, 0)
        self.straight_cam_orient = np.matrix([
            [1, 0, 0],  # Right side of the camera
            [0, 0, 1],  # Top of the camera
            [0, 1, 0]   # Optical axis
        ])

    def test_all_corners_in_front(self):
        # Define the camera position to be in front of the square
        cam_pos = np.array([0, -15, 0])

        new_corners, lines, factors = get_cutoff_points(self.flat_square, cam_pos, self.straight_cam_orient)

        self.assertTrue(np.array_equal(new_corners, self.flat_square))
        self.assertEqual(lines, [0, 1, 2, 3])
        self.assertEqual(factors, [0, 0, 0, 0])

    def test_one_corner_behind(self):
        """
        The camera is positioned just in front of C1 and yaw'ed 45 degrees. This puts
        C1 behind the camera.
        """
        cam_pos = np.array([-9, -10, 0])

        # Yaw -45 degrees
        cam_orient = np.matrix([
            [1, -1, 0],  # Right side of the camera
            [0, 0, 1],   # Top of the camera
            [1, 1, 0]    # Optical axis
        ])

        new_corners, lines, factors = get_cutoff_points(self.flat_square, cam_pos, cam_orient)
        self.assertEqual(len(new_corners), 5)
        self.assertTrue(np.array_equal(new_corners, np.array([
            [-10, -9, 0],
            self.flat_square[1],
            self.flat_square[2],
            self.flat_square[3],
            [-9, -10, 0]
        ])))
        self.assertEqual(lines, [0, 1, 2, 3, 3])
        self.assertEqual(factors, [0.05, 0, 0, 0, 0.95])

    def test_two_corners_behind(self):
        """
        Set the camera in the middle of the square and point it straight forward. This
        will cut the square in half.
        """
        # Position camera at origin
        cam_pos = np.array([0, 0, 0])

        new_corners, lines, factors = get_cutoff_points(self.flat_square, cam_pos, self.straight_cam_orient)
        self.assertEqual(len(new_corners), 4)
        self.assertTrue(np.array_equal(new_corners, np.array([
            [-10, 0, 0],
            self.flat_square[1],
            self.flat_square[2],
            [10, 0, 0]
        ])))
        self.assertEqual(lines, [0, 1, 2, 2])
        self.assertEqual(factors, [0.5, 0, 0, 0.5])

    def test_three_corners_behind(self):
        """
        Set the camera on the border between C1 and C4 and turn it away from the origin.
        This will make all corners except for C1 be behind the camera.
        """
        cam_pos = np.array([-9, -10, 0])

        # Yaw 135 degrees
        cam_orient = np.matrix([
            [-1, 1, 0],  # Right side of the camera
            [0, 0, 1],   # Top of the camera
            [-1, -1, 0]  # Optical axis
        ])

        new_corners, lines, factors = get_cutoff_points(self.flat_square, cam_pos, cam_orient)
        self.assertEqual(len(new_corners), 3)
        self.assertTrue(np.array_equal(new_corners, np.array([
            self.flat_square[0],
            [-10, -9, 0],   # New corner
            [-9, -10, 0]    # New corner
        ])))
        self.assertEqual(lines, [0, 0, 3])
        self.assertEqual(factors, [0, 0.05, 0.95])

    def test_cut_directly_in_corner(self):
        """
        Corner case (pun intended): Put camera in origin and point it directly at C3.
        This will make the camera plane cut directly through C2 and C4.
        """
        # Position camera at origin
        cam_pos = np.array([0, 0, 0])

        # Yaw -45 degrees
        cam_orient = np.matrix([
            [1, -1, 0],  # Right side of the camera
            [0, 0, 1],   # Top of the camera
            [1, 1, 0]    # Optical axis
        ])

        new_corners, lines, factors = get_cutoff_points(self.flat_square, cam_pos, cam_orient)
        self.assertEqual(len(new_corners), 3)
        self.assertTrue(np.array_equal(new_corners, np.array([
            self.flat_square[1],
            self.flat_square[2],
            self.flat_square[3]
        ])))
        self.assertEqual(lines, [1, 2, 3])
        self.assertEqual(factors, [0, 0, 0])


class TestCutoffTextures(unittest.TestCase):
    def test_cut_texture(self):
        # Mock output from the get_cutoff_points function
        lines = [0, 1, 2, 3, 3]
        factors = [0.05, 0, 0, 0, 0.95]

        # The texture polygon
        texture = np.array([
            [-5, -5],
            [-5, 5],
            [5, 5],
            [5, -5]
        ])

        # Find new corners and assert correctness
        new_corners = get_corners_of_cut_texture(texture, lines, factors)
        self.assertEqual(len(new_corners), 5)
        self.assertTrue(np.array_equal(new_corners, np.array([
            [-5, -4.5],
            texture[1],
            texture[2],
            texture[3],
            [-4.5, -5]
        ])))


class TestAddDummyPoint(unittest.TestCase):
    def setUp(self):
        self.c1 = [-10, -10, 0]
        self.c2 = [-10, 10, 0]
        self.c3 = [10, 10, 0]
        self.c4 = [10, -10, 0]

    def test_add_point_free_line(self):
        # Mock output from the get_cutoff_points function
        lines = [0, 1, 2]
        factors = np.array([0, 0, 0], dtype=np.float32)
        cutoff_polygon = np.array([self.c1, self.c2, self.c3])

        # Find new corners and assert correctness
        corners, lines, factors = add_dummy_point(cutoff_polygon, lines, factors)
        self.assertTrue(np.array_equal(corners, np.array([self.c1, [-10, 0, 0], self.c2, self.c3])))
        self.assertTrue(np.array_equal(lines, [0, 0, 1, 2]))
        self.assertAlmostEqual(factors[1], 0.5)

    def test_first_point_is_not_in_corner(self):
        # Mock output from the get_cutoff_points function
        lines = [0, 1, 2]
        factors = np.array([0.1, 0, 0], dtype=np.float32)
        cutoff_polygon = np.array([[-10, -8, 0], self.c2, self.c3])

        # Find new corners and assert correctness
        corners, lines, factors = add_dummy_point(cutoff_polygon, lines, factors)
        self.assertTrue(np.array_equal(corners, np.array([[-10, -8, 0], [-10, 1, 0], self.c2, self.c3])))
        self.assertTrue(np.array_equal(lines, [0, 0, 1, 2]))
        self.assertAlmostEqual(factors[1], 0.55)

    def test_second_point_is_not_in_corner(self):
        # Mock output from the get_cutoff_points function
        lines = [0, 0, 3]
        factors = np.array([0, 0.4, 0], dtype=np.float32)
        cutoff_polygon = np.array([self.c1, [-10, -2, 0], self.c4])

        # Find new corners and assert correctness
        corners, lines, factors = add_dummy_point(cutoff_polygon, lines, factors)
        self.assertTrue(np.array_equal(corners, np.array([self.c1, [-10, -6, 0], [-10, -2, 0], self.c4])))
        self.assertTrue(np.array_equal(lines, [0, 0, 0, 3]))
        self.assertAlmostEqual(factors[1], 0.2)

    def test_no_lines_on_line_zero(self):
        # Mock output from the get_cutoff_points function
        lines = [1, 2, 3]
        factors = np.array([0, 0, 0], dtype=np.float32)
        cutoff_polygon = np.array([self.c2, self.c3, self.c4])

        # Find new corners and assert correctness
        corners, lines, factors = add_dummy_point(cutoff_polygon, lines, factors)
        self.assertTrue(np.array_equal(corners, np.array([self.c2, [0, 10, 0], self.c3, self.c4])))
        self.assertTrue(np.array_equal(lines, [1, 1, 2, 3]))
        self.assertAlmostEqual(factors[1], 0.5)


class TestModelDistanceComparator(unittest.TestCase):
    def setUp(self):
        # Define the orientation to be looking straight at (0, 0, 0)
        self.straight_cam_orient = np.matrix([
            [1, 0, 0],  # Right side of the camera
            [0, 0, 1],  # Top of the camera
            [0, 1, 0]   # Optical axis
        ])

        self.origin_cam_pos = np.array([0, 0, 0])
        self.comparator = get_model_comparator(self.origin_cam_pos, self.straight_cam_orient)

    def test_behind_each_other(self):
        # Square directly in front of the camera 1 unit away
        o1 = np.array([
            [-1, 1, 1],
            [1, 1, 1],
            [1, 1, -1],
            [-1, 1, -1],
        ])

        # Square directly in front of the camera 2 units away
        o2 = np.array([
            [-1, 2, 1],
            [1, 2, 1],
            [1, 2, -1],
            [-1, 2, -1],
        ])

        self.assertLess(self.comparator(o1, o2), 0)
        self.assertGreater(self.comparator(o2, o1), 0)
        self.assertEqual(self.comparator(o1, o1), 0)

    def test_side_by_side(self):
        # Square directly in front of the camera 1 unit away
        o1 = np.array([
            [-1, 1, 1],
            [1, 1, 1],
            [1, 1, -1],
            [-1, 1, -1],
        ])

        # o1 shifted along the x axis
        o2 = np.array([
            [2, 1, 1],
            [4, 1, 1],
            [4, 1, -1],
            [2, 1, -1],
        ])

        self.assertLess(self.comparator(o1, o2), 0)


if __name__ == '__main__':
    unittest.main()
