import sys


def deep_greedy_insertion(schedule: dict, visits_pool: set):
    while visits_pool:
        best_voyage = [-1, None]
        for day, visit in visits_pool:
            best_cost = sys.maxsize
            voyages = get_voyages_by_departure_day(schedule, day)
            for voyage in voyages:
                distance, route = solve_tsp(voyage, visit)
                if calc_variable_cost(distance) - voyage.variable_cost < best_cost:
                    best_voyage = [day, voyage]
        best_voyage[1].add_node(visit)
        visits_pool.remove(best_voyage)


def k_regret_insertion(schedule: dict, visits_pool: set, k=2):
    while visits_pool:
        best_visit = None
        best_voyage = None
        best_regret_value = 0
        for day, visit in visits_pool:
            best_local_voyage = None
            voyages = get_voyages_by_departure_day(schedule, day)
            diff_cost = [[0, None] for _ in range(k)]
            for voyage in voyages:
                distance, route = solve_tsp(voyage, visit)
                insertion_cost = calc_variable_cost(distance) - voyage.variable_cost
                for i, value in diff_cost:
                    if value > insertion_cost:
                        t = diff_cost[i]
                        diff_cost[i] = [insertion_cost, voyage]
                        insertion_cost = t
            best_local_voyage = diff_cost[0][1]
            if diff_cost[k - 1][0] == 0:  # that means there is no k-best voyage to insert the visit
                regret_value = 0
            else:
                regret_value = diff_cost[0][0] - diff_cost[k - 1][0]
            if regret_value > best_regret_value:
                best_regret_value = regret_value
                best_visit = visit
                best_voyage = best_local_voyage
        best_voyage[1].add_node(best_visit)
        visits_pool.remove(best_visit)
