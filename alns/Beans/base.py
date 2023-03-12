from alns.utils.coord import Coord


class Base:
    def __init__(self,
                 name: str,
                 service_time: float,
                 time_window: list,
                 longitude: float,
                 latitude: float):
        self.name = name
        self.service_time = service_time
        self.time_window = time_window
        self.longitude = longitude
        self.latitude = latitude
        self.location = Coord(longitude, latitude)

    def __repr__(self):
        return f'Base : {self.name}, location: {self.location}'
