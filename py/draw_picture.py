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
    :param points: four points in 3D space defining a rectangle
    :param cam_pos: the position of the camera
    :param cam_orient: the orientation of the camera
    :return: the points where the rectangle cuts the image plane if any
    """
    # Check if any points are behind the camera
    def is_behind_camera(point):
        vector_point_camera = point - cam_pos
        in_angle = angle(vector_point_camera, cam_orient[2].getA()[0])
        return in_angle > math.pi / 2

    point_behind_camera = [is_behind_camera(p) for p in points]
    if not any(point_behind_camera):
        # No points are behind the camera
        return []

    # One or more corners are behind the camera check all lines
    intersections = []
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        # If the current point are in front of the camera and the next is behind the line is cut
        if point_behind_camera[i] is not point_behind_camera[next_i]:
            # Calculate where the line between the points intersects with the camera plane
            line_direction = points[next_i] - points[i]
            nominator = dotproduct(cam_pos - points[i], cam_orient[2].getA1())
            denominator = dotproduct(line_direction, cam_orient[2].getA1())
            factor = float(nominator) / denominator
            intersection = (factor * line_direction + points[i])
            intersections.append(intersection)

    return intersections

if __name__ == '__main__':
    main()