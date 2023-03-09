import numpy as np


class DistanceManager:
    def __init__(self,
                 base,
                 insts):
        self.base = base
        self.insts = insts
        self.nodes = [base] + self.insts
        self.mapping = self.name_mapping()
        self.distance_matrix = self.calc_distance_matrix()

    def name_mapping(self):
        mapping = {}
        for i, node in enumerate(self.nodes):
            mapping[node.name] = i
        return mapping

    def calc_distance_matrix(self):
        locations = [node.location for node in self.nodes]
        distances = [[i.geo_distance_to_coord(j) for j in locations] for i in locations]
        return np.array(distances)

    def distance(self, from_node, to_node):
        return self.distance_matrix[self.mapping[from_node], self.mapping[to_node]]