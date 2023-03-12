from alns.Beans.installation import Installation


class Route:
    def __init__(self,
                 base,
                 inst_seq,
                 distance_manager):
        self.route = [base] + inst_seq + [base]
        self.distance_manager = distance_manager

    def edges(self):
        """
        Edges of the route
        :return: list of edges [from_node, to_node, distance]
        :rtype: list[list[Installation, Installation, float]]
        """
        edges = zip(self.route[:-1], self.route[1:])
        return [[from_node, to_node, self.distance_manager.distance(from_node.name, to_node.name)]
                for (from_node, to_node) in edges]

    def __repr__(self):
        return f'{[x.name for x in self.route[1:-1]]}'
