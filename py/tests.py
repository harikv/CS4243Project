import unittest

import numpy as np
from draw_picture import get_cutoff_points, get_corners_of_cut_texture


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
            [-10, -9, 0],   # New corner
            [-10, 10, 0],   # Corner C2
            [10, 10, 0],    # Corner C3
            [10, -10, 0],   # Corner C4
            [-9, -10, 0]    # New corner
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
            [-10, 0, 0],    # New corner
            [-10, 10, 0],   # Corner C2
            [10, 10, 0],    # Corner C3
            [10, 0, 0]      # New corner
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
            [-10, -10, 0],  # Corner C1
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
            [-10, 10, 0],   # Corner C2
            [10, 10, 0],    # Corner C3
            [10, -10, 0]    # Corner C4
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
            [-5, -5, 0],  # Corner 1
            [-5, 5, 0],   # Corner 2
            [5, 5, 0],    # Corner 3
            [5, -5, 0]    # Corner 4
        ])

        # Find new corners and assert correctness
        new_corners = get_corners_of_cut_texture(texture, lines, factors)
        self.assertEqual(len(new_corners), 5)
        self.assertTrue(np.array_equal(new_corners, np.array([
            [-5, -4.5, 0],  # Corner
            [-5, 5, 0],     # Corner 2
            [5, 5, 0],      # Corner 3
            [5, -5, 0],     # Corner 4
            [-4.5, -5, 0]   # Corner
        ])))


if __name__ == '__main__':
    unittest.main()
