from projection import angle, dotproduct, length
import math
import numpy as np


def get_cutoff_points(points, cam_pos, cam_orient):
    """
    If some of the points are behind the camera, this function finds out where the
    image plane cuts the object.
    :param points: four points in 3D space defining a polygon
    :param cam_pos: the position of the camera
    :param cam_orient: the orientation of the camera
    :return: a 3-tuple containing:
               the corners of the sliced polygon
               which line in the original polygon the corner lies on
               a factor telling how far up the line the new corner is located

    """
    # Check if any points are behind the camera
    def is_behind_camera(point):
        vector_point_camera = point - cam_pos
        in_angle = angle(vector_point_camera, cam_orient[2].getA()[0])
        return in_angle > math.pi / 2

    point_behind_camera = [is_behind_camera(p) for p in points]

    # One or more corners are behind the camera check all lines
    new_corners = []
    line_segments = []
    factors = []
    for i in range(len(points)):
        next_i = (i + 1) % len(points)

        # Add corner if it is not behind the camera
        if not point_behind_camera[i]:
            new_corners.append(points[i])
            line_segments.append(i)
            factors.append(0)

        # If the current point are in front of the camera and the next is behind the line is cut
        if point_behind_camera[i] is not point_behind_camera[next_i]:
            # Calculate where the line between the points intersects with the camera plane
            line_direction = points[next_i] - points[i]
            nominator = dotproduct(cam_pos - points[i], cam_orient[2].getA1())
            denominator = dotproduct(line_direction, cam_orient[2].getA1())
            factor = float(nominator) / denominator
            intersection = (factor * line_direction + points[i])

            # If the factor is 0 or 1, the camera plane cuts directly through an existing corner.
            # This will be added in the next iteration, so don't add it here.
            if 0 < factor < 1:
                new_corners.append(intersection)
                line_segments.append(i)
                factors.append(factor)

    return new_corners, line_segments, factors


def add_dummy_point(points, lines, factors):
    """
    Add a point to the polygon by putting in a dummy point on the first line
    :param points: corner points of the polygon
    :param lines: indices of the lines to cut
    :param factors: how far up the line a cut should be made (one factor pr. line)
    :return: a similar polygon with an extra point inserted on the first line.
    """
    new_line = lines[0]
    new_factor = (1 + factors[0]) * 0.5 if lines[0] != lines[1] else factors[1] * 0.5
    new_point = points[0] + (points[1] - points[0]) * 0.5

    return np.insert(points, 1, new_point, axis=0), \
           np.insert(lines, 1, new_line, axis=0), \
           np.insert(factors, 1, new_factor, axis=0)


def get_corners_of_cut_texture(points, lines, factors):
    """
    Find the new corners of a polygon given a list of line indices and factors
    describing how far up the line to put the new corner.
    :param points: corner points of the polygon
    :param lines: indices of the lines to cut
    :param factors: how far up the line a cut should be made (one factor pr. line)
    :return: the corners of the cut polygon
    """

    def cut_line(info):
        next_i = (info[0] + 1) % len(points)
        line_direction = points[next_i] - points[info[0]]
        return info[1] * line_direction + points[info[0]]

    return [cut_line(x) for x in zip(lines, factors)]


def get_optical_axis(cam_orient):
    return cam_orient[2].getA1()


def get_model_comparator(cam_pos, cam_orient):
    """
    Returns a comparison function that takes two 3D polygons and returns a number
    less than zero if the first parameter is closer to the camera and a number greater
    than zero if the second parameter is the closest.
    It is determined by projecting the middle of the object onto the optical axis and
    determining which projection is the longest. If they are equally long, the Euclidian
    distance between the camera and the center points are compared.
    If the objects are equally far away, zero is returned.
    :param cam_pos: the position of the camera
    :param cam_orient: camera orientation
    :return: object comparator function
    """
    def comp(o1, o2):
        """
        Compare two objects according to the description above.
        :param o1: first polygon
        :param o2: second polygon
        :return: -x when o1 is closer than o2 to the camera, x when o1 is further away
                 than o2, zero if they are equally far away.
        """
        # Find center of polygons
        o1_center = sum(o1) / float(len(o1))
        o2_center = sum(o2) / float(len(o2))

        optical_axis = get_optical_axis(cam_orient)
        oa_dot_oa = dotproduct(optical_axis, optical_axis)

        # Calculate the distance in the direction of the optical axis
        vec_to_o1 = o1_center - cam_pos
        vec_to_o2 = o2_center - cam_pos

        # Project points onto the optical axis
        proj_dist_to_o1 = length((dotproduct(vec_to_o1, optical_axis) / oa_dot_oa) * optical_axis)
        proj_dist_to_o2 = length((dotproduct(vec_to_o2, optical_axis) / oa_dot_oa) * optical_axis)

        # Compare projected distances
        if proj_dist_to_o1 != proj_dist_to_o2:
            return proj_dist_to_o1 - proj_dist_to_o2

        # Compare euclidian distances
        dist_to_o1 = length(o1_center - cam_pos)
        dist_to_o2 = length(o2_center - cam_pos)
        return dist_to_o1 - dist_to_o2

    return comp