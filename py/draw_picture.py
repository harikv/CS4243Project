from projection import project, angle, dotproduct
from geometry import Vector, Rectangle, Triangle
import math

viewing_angle_in_radians = math.pi/2

def main():
    # First try to draw scene
    shapes = [
        Rectangle(Vector(-10, -30, -3), Vector(0, 30, 0),   Vector(20, 0, 0)),  # Grass
        Triangle(Vector(-4, 0, 1.5),    Vector(4, 0, 1.5),  Vector(8, 0, 0)),   # Church front facade top
        Rectangle(Vector(-4, 0, -2.5),  Vector(0, 0, 4),    Vector(0, 5, 0)),   # Church left side of front facade
        Rectangle(Vector(4, 0, -2.5),   Vector(0, 0, 4),    Vector(0, 5, 0)),   # Church right side of front facade
        Rectangle(Vector(-4, 0, 1.5),   Vector(4, 0, 1.5),  Vector(0, 5, 0)),   # Church left side of ceiling
        Rectangle(Vector(0, 0, 3),      Vector(4, 0, -1.5), Vector(0, 5, 0))    # Church right side of ceiling
    ]

    church_front_facade = Rectangle(Vector(-4, 0, -2.5), Vector(0, 0, 4), Vector(8, 0, 0))  # Church front facade
    church_front_facade.punch_hole(('rect', 0, 3, 3, 2))
    church_front_facade.punch_hole(('circ', 3, 4, 1))
    shapes.append(church_front_facade)

    pts = []
    for x in [s.get_points(ppm=3) for s in shapes]:
        pts.extend(x)


    project(pts, [0, -20, 1.5], viewing_angle_in_radians)


def get_cutoff_points(points, cam_pos, cam_orient):
    """
    If some of the points are behind the camera, this function finds out where the
    image plane cuts the object.
    :param points: four points in 3D space defining a polygon
    :param cam_pos: the position of the camera
    :param cam_orient: the orientation of the camera
    :return: a 3-tuple containing:
               the corners of the sliced polygon
               which line in the original polygon the corner lies on (-1 for existing corners)
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
            line_segments.append(-1)
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

if __name__ == '__main__':
    main()