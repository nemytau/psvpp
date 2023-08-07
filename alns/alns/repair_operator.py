from alns.alns.mutation_service import added_costs_for_visits_insertion, min_insertion_added_cost_index


def deep_greedy_insertion(visit_pool, sch):
    """
    Insert visit into schedule sch using deep greedy insertion.
    :param visit_pool:
    :type visit_pool: list[Visit]
    :param sch:
    :return:
    """
    visit_added_costs = added_costs_for_visits_insertion(visit_pool, sch)
    for visit, added_costs in visit_added_costs:
        print(visit, added_costs)
    print('---')
    while len(visit_added_costs) > 0:
        min_idx = min_insertion_added_cost_index(visit_added_costs)
        # TODO: Check why id:7 has added cost inf
        print(min_idx)
        print(visit_added_costs[min_idx])
        break

def k_regret_insertion():
    pass

