from alns.alns.mutation_service import *
import numpy as np


def deep_greedy_insertion(inst_pool, sch):
    """
    Insert visit into schedule sch using deep greedy insertion.
    :param inst_pool:
    :type inst_pool: list[Visit]
    :param sch:
    :return:
    """
    # TODO: Check for duplicates of visits in visit_pool and duplicate them in added costs
    if not inst_pool:
        return True
    inst_added_costs = added_costs_for_visits_insertion(inst_pool, sch)
    # for inst, added_costs in inst_added_costs:
    #     print(inst, added_costs)
    # print('---')
    while len(inst_added_costs) > 0:
        min_idx = min_insertion_added_cost_index(inst_added_costs)
        inst = inst_added_costs[min_idx][0]
        voyage = inst_added_costs[min_idx][1][0][0]
        min_cost = inst_added_costs[min_idx][1][0][1]
        if min_cost == np.inf:
            return False
        voyage.insert_visit(inst)
        inst_added_costs = update_insertion_added_costs(inst_added_costs, min_idx, sch)
    return True


def k_regret_insertion(inst_pool, sch, k=2):
    """
    Insert visit into schedule sch using k-regret insertion.
    :param inst_pool:
    :type inst_pool: list[Visit]
    :param sch:
    :return:
    """
    if not inst_pool:
        return True
    inst_added_costs = added_costs_for_visits_insertion(inst_pool, sch)
    # for inst, added_costs in inst_added_costs:
    #     print(inst, added_costs)
    # print('---')
    while len(inst_added_costs) > 0:
        min_idx = min_kregret_insertion_added_cost_index(inst_added_costs, k)
        inst = inst_added_costs[min_idx][0]
        voyage = inst_added_costs[min_idx][1][0][0]
        min_cost = inst_added_costs[min_idx][1][0][1]
        if min_cost == np.inf:
            return False
        voyage.insert_visit(inst)
        inst_added_costs = update_insertion_added_costs(inst_added_costs, min_idx, sch)
    return True

