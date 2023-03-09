import numpy as np
from haversine import haversine, Unit


class Coord:
    def __init__(self, x, y):
        """
        Create Coord instance.

        :param x: latitude.
        :type x: float
        :param y: longitude.
        :type y: float
        """
        self.x = x
        self.y = y
        self.lat = x
        self.lon = y
        self.coord = (x, y)
        self.rad_coord = tuple(np.radians(self.coord))

    @classmethod
    def from_pair(cls, coords):
        """
        Create Coord from tuple or list.

        :param coords: longitude and latitude tuple.
        :type coords: tuple[float, float], list[float]
        """
        if len(coords) != 2:
            raise ValueError('Coordinate should consist of 2 values: longitude and latitude.')
        return cls(*coords)

    @staticmethod
    def geo_distance(coord1, coord2, unit='nmi'):
        """
        Calculate geodesic distance between two points.

        :param coord1: tuple of coordinates (lat, lon) of start point in decimal degrees.
        :type coord1: tuple[float, float]
        :param coord2: tuple of coordinates (lat, lon) of end point in decimal degrees.
        :type coord2: tuple[float, float]
        :param unit: unit name from ['nmi', 'km'], defaults to 'nmi'.
        :type unit: str
        :return: distance in selected units.
        :rtype: float
        """
        if unit == 'nmi':
            distance = haversine(coord1, coord2, unit=Unit.NAUTICAL_MILES)
        elif unit == 'km':
            distance = haversine(coord1, coord2, unit=Unit.KILOMETERS)
        else:
            raise ValueError('Wrong unit string')
        return distance

    def geo_distance_to_coord(self, other, unit='nmi'):
        """
        Calculate geodesic distance to other point.

        :param other: Coord instance.
        :type other: Coord
        :param unit: unit name from ['nmi', 'km'], defaults to 'nmi'.
        :type unit: str
        :return: distance in selected units.
        :rtype: float
        """
        distance = self.geo_distance(self.coord, other.coord, unit)
        return distance

    def __repr__(self):
        return f'[{self.x}, {self.y}]'
