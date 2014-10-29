from projection import project
from geometry import Vector, Rectangle, Triangle


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
    for x in [s.get_points(ppm=1.5) for s in shapes]:
        pts.extend(x)

    project(pts, [0, -30, 1.5])


if __name__ == '__main__':
    main()