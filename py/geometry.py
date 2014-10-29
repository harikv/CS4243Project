import numpy as np
import math


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.length = math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def scale(self, factor):
        return Vector(self.x * factor, self.y * factor, self.z * factor)

    def len(self):
        return self.length

    def to_tuple(self):
        return self.x, self.y, self.z

    def dist(self, other):
        return (self - other).len()

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)


class Shape(object):
    def __init__(self, side1, side2):
        self._holes = []
        self._side1 = side1
        self._side2 = side2

    def _in_hole(self, v1, v2, newvec):
        for hole in self._holes:
            if hole[0] == 'rect':
                if hole[1] < v1 < hole[1] + hole[3] and hole[2] < v2 < hole[2] + hole[4]:
                    return True
            if hole[0] == 'circ':
                center = self._side1.scale(hole[1] / self._side1.len()) + self._side2.scale(hole[2] / self._side2.len())
                if center.dist(newvec) < hole[3]:
                    return True
        return False

    def punch_hole(self, shape):
        self._holes.append(shape)


class Rectangle(Shape):
    def __init__(self, basevec, side1, side2):
        Shape.__init__(self, side1, side2)
        self.baseVec = basevec

    def get_points(self, ppm=1):
        pts = []

        # TODO get some heavy optimization going on here
        for i in np.arange(0.0, self._side1.len(), 1.0 / ppm):
            for j in np.arange(0.0, self._side2.len(), 1.0 / ppm):
                newvec = self._side1.scale(i / self._side1.len()) + self._side2.scale(j / self._side2.len())
                newpoint = self.baseVec + newvec

                # Check if we are in a hole
                if not self._in_hole(i, j, newvec):
                    pts.append(newpoint.to_tuple())

        return pts


class Triangle(Shape):
    def __init__(self, basevec, side1, side2):
        Shape.__init__(self, side1, side2)
        self.baseVec = basevec

    def get_points(self, ppm=1):
        pts = []

        for i in np.arange(0.0, self._side1.len(), 1.0 / ppm):
            normalized = i / self._side1.len()
            for j in np.arange(0.0, self._side2.len() * (1 - normalized), 1.0 / ppm):
                newvec = self._side1.scale(i / self._side1.len()) + self._side2.scale(j / self._side2.len())
                newpoint = self.baseVec + newvec

                # Check if we are in a hole
                if not self._in_hole(i, j, newvec):
                    pts.append(newpoint.to_tuple())

        return pts