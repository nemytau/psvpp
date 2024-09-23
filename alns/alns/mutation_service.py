from itertools import combinations

import numpy as np

from alns.Beans.schedule import Schedule


def removal_added_cost(inst, voyage):
    """
    Calculates added cost of having visit in the schedule  sch.
    :param inst:
    :type inst: Installation
    :param voyage:
    :type voyage: Voyage
    :param sch:
    :type sch: Schedule
    :return:
    :rtype: float
    """
    new_voyage = voyage.__deepcopy__()
    new_voyage.route = [i for i in new_voyage.route if i != inst]
    new_voyage.improve_full_enum()
    old_cost = voyage.calc_variable_cost()
    new_cost = new_voyage.calc_variable_cost()
    return new_cost - old_cost


def insertion_added_cost(sch, inst, target):
    """
    Calculates added cost of inserting installation visit in the schedule sch.
    :param target:
    :type target: Voyage
    :param sch:
    :type sch: Schedule
    :param inst:
    :type inst: Installation
    :return:
    :rtype: float|bool
    """
    if inst in target.route:
        return np.inf
    new_voyage = target.__deepcopy__()
    new_voyage.add_inst(inst)
    new_voyage.improve_full_enum()
    if sch.check_for_replacement_overlap(new_voyage, target):
        return np.inf
    old_cost = target.calc_variable_cost()
    new_cost = new_voyage.calc_variable_cost()
    if sch.is_vessel_idle(target.vessel):
        new_cost = new_cost + target.vessel.cost
    return new_cost - old_cost


def is_departure_spread_ok(sch, inst, origin_voyage, target_voyage):
    """
    :param sch:
    :param inst:
    :param target_voyage:
    :param origin_voyage: if None, inst is not in any voyage
    :return:
    """
    departure_spread_is_ok = True
    for voyage in sch.find_voyages_containing_visit(inst):
        if (voyage == target_voyage) or (voyage == origin_voyage):
            continue
        if abs(voyage.start_day - target_voyage.start_day) <= inst.departure_spread:
            departure_spread_is_ok = False
            break
    return departure_spread_is_ok


def is_enough_space_on_deck(inst, target_voyage):
    """
    :param inst:
    :param target_voyage:
    :return:
    """
    return target_voyage.vessel.deck_capacity >= target_voyage.deck_load + inst.deck_demand


def is_insertion_feasible(sch, inst, target):
    """
    Checks if insertion of visit in voyage is feasible.
    :param inst:
    :param target:
    :param sch:
    :return:
    :rtype: bool
    """
    return is_relocation_feasible(sch, inst, None, target)


def is_relocation_feasible(sch, inst, origin, target):
    visit_is_duplicated = inst in target.route
    # TODO: transfer implementation to Voyage class?
    space_enough = is_enough_space_on_deck(inst, target)
    # TODO: transfer implementation to Schedule class?
    departure_spread_is_ok = is_departure_spread_ok(sch, inst, origin, target)
    return not visit_is_duplicated and departure_spread_is_ok and space_enough


def _sort_visit_added_costs(visit_added_costs):
    return sorted(visit_added_costs, key=lambda x: x[1], reverse=False)


def added_costs_for_visits_removal(inst_voyage_list, sch):
    added_costs = []
    for (inst, voyage) in inst_voyage_list:
        added_costs.append(removal_added_cost(inst, voyage))
    visit_added_costs = list(zip(inst_voyage_list, added_costs))
    visit_added_costs = _sort_visit_added_costs(visit_added_costs)
    return visit_added_costs


def added_costs_for_visits_insertion(insts, sch):
    """
    Calculates added cost matrix of inserting insts as visits into schedule sch.
    The matrix is sorted by added cost ascending for each inst.
    :param insts:
    :type insts: list[Installation]
    :param sch:
    :type sch: Schedule
    :param ignore_infeasibility: if True, added costs for infeasible insertions are not np.inf
    :type ignore_infeasibility: bool
    :return:
    """
    added_costs = []
    voyages = [voyage for vessel_voyages in sch.schedule.values() for voyage in vessel_voyages]
    for inst in insts:
        visit_added_costs = []
        for voyage in voyages:
            if is_insertion_feasible(sch, inst, voyage):
                visit_added_costs.append((voyage, insertion_added_cost(sch, inst, voyage)))
            else:
                visit_added_costs.append((voyage, np.inf))
        added_costs.append(_sort_visit_added_costs(visit_added_costs))
    return list(zip(insts, added_costs))


def update_removal_added_costs(visit_added_costs, removed_inst, origin_voyage, sch):
    """
    Updates the added costs of visits in visit_added_costs after removal of removed_visit.
    :param visit_added_costs:
    :type visit_added_costs: list[tuple(Visit, float)]
    :param removed_inst:
    :type removed_inst: Installation
    :param origin_voyage:
    :type origin_voyage: Voyage
    :param sch:
    :type sch: Schedule
    :return:
    """
    visit_added_costs_up_to_date = [((inst, voyage), acost) for ((inst, voyage), acost) in visit_added_costs
                                    if not (voyage == origin_voyage)]
    visits_added_cost_to_update = [(inst, voyage) for ((inst, voyage), acost) in visit_added_costs
                                   if (voyage == origin_voyage) & (not inst == removed_inst)]
    updated_visit_added_costs = added_costs_for_visits_removal(visits_added_cost_to_update, sch)
    return _sort_visit_added_costs(visit_added_costs_up_to_date + updated_visit_added_costs)


def update_insertion_added_costs(visit_added_costs, inserted_inst_idx, sch):
    """
    Updates the added costs of visits in visit_added_costs after insertion of inserted_visit.
    :param visit_added_costs:
    :type visit_added_costs: list[tuple(Visit, float)]
    :param inserted_inst_idx:
    :type inserted_inst_idx: int
    :param sch:
    :type sch: Schedule
    :return:
    """

    def same_vessel(x, y):
        return x.vessel == y.vessel

    # For same vessel: recalculate added costs for all non-empty voyages for all insts
    # For another vessels: check for departure spread and set inf if not ok
    target_voyage = visit_added_costs[inserted_inst_idx][1][0][0]
    updated_added_costs = []
    for (inst, acosts) in visit_added_costs[:inserted_inst_idx] + visit_added_costs[inserted_inst_idx + 1:]:
        same_vessel_acosts = [(voyage, cost) for (voyage, cost) in acosts
                              if same_vessel(voyage, target_voyage)]
        another_vessel_acosts = [(voyage, cost) for (voyage, cost) in acosts
                                 if not same_vessel(voyage, target_voyage)]
        updated_inst_acosts = []
        for voyage, cost in another_vessel_acosts:
            if not is_departure_spread_ok(sch, inst, None, voyage):
                updated_inst_acosts.append((voyage, np.inf))
            else:
                updated_inst_acosts.append((voyage, cost))
        for voyage, cost in same_vessel_acosts:
            if is_insertion_feasible(sch, inst, voyage):
                updated_inst_acosts.append((voyage, insertion_added_cost(sch, inst, voyage)))
            else:
                updated_inst_acosts.append((voyage, np.inf))

        updated_added_costs.append((inst, _sort_visit_added_costs(updated_inst_acosts)))
    return updated_added_costs


def min_insertion_added_cost_index(visit_added_costs):
    """
    :param visit_added_costs:
    :type visit_added_costs: list[tuple(Visit, list[tuple(Voyage, float)])]
    :return:
    """
    min_added_cost = np.inf
    min_added_cost_index = -1
    for i, (inst, voyage_added_costs) in enumerate(visit_added_costs):
        for voyage, added_cost in voyage_added_costs:
            if added_cost < min_added_cost:
                min_added_cost = added_cost
                min_added_cost_index = i
    return min_added_cost_index


def min_kregret_insertion_added_cost_index(visit_added_costs, k):
    """
    :param visit_added_costs:
    :type visit_added_costs: list[tuple(Visit, list[tuple(Voyage, float)])]
    :param k:
    :type k: int
    :return:
    """
    min_added_cost = np.inf
    min_added_cost_index = -1
    for i, (inst, voyage_added_costs) in enumerate(visit_added_costs):
        voyage_added_costs = sorted(voyage_added_costs, key=lambda x: x[1])
        added_cost = np.inf
        if len(voyage_added_costs) == 1:
            # NOTE: not necessary to check for feasibility here, it is probably not possible
            # and case covered by setting added_cost to np.inf
            raise ValueError('Only one voyage available for insertion')
        elif 1 < len(voyage_added_costs) < k:
            added_cost = voyage_added_costs[0][1] - voyage_added_costs[-1][1]
        elif len(voyage_added_costs) >= k:
            added_cost = voyage_added_costs[0][1] - voyage_added_costs[k - 1][1]
        if (added_cost < min_added_cost) and voyage_added_costs[0][1] < np.inf:
            min_added_cost = added_cost
            min_added_cost_index = i
    return min_added_cost_index


def relocation_added_costs_from_insertion_and_removal(insertion_added_costs, removal_added_costs):
    """
    Calculates added cost matrix of relocating visits from one voyage to another one.
    :param insertion_added_costs:
    :param removal_added_costs:
    :return:
    """
    relocation_added_costs = []
    # Не уверен, что сработает
    insertion_added_costs_dict = dict(insertion_added_costs)
    for ((inst, from_voyage), removal_acost) in removal_added_costs:
        # List of tuples (voyage, acost)
        inst_acosts = insertion_added_costs_dict[inst]
        for to_voyage, acost in inst_acosts:
            relocation_added_costs.append((inst, from_voyage, to_voyage, acost + removal_acost))
    relocation_added_costs = sorted(relocation_added_costs, key=lambda x: x[3])
    return relocation_added_costs


def relocation_added_cost(sch, inst, origin, target):
    """
    Calculates added cost of relocating visit from origin to target voyage. It is assumed, that relocation is feasible.
    :param sch:
    :param inst:
    :param origin:
    :param target:
    :return:
    """
    # TODO: check if its correct
    # Seems like it is correct. Since this method is called if relocation is feasible, than relocation cost is removal
    # cost + insertion cost.
    rcost = removal_added_cost(inst, origin)
    icost = insertion_added_cost(sch, inst, target)
    return rcost + icost


# TODO: consider adding list of visits to calculate added costs for relocation
# It is actually not necessary, since it is always used for all visits
def added_costs_for_visits_relocation(sch):
    """
    Calculates added cost matrix of relocating visits in the schedule sch.
    :param sch:
    :type sch: Schedule
    :return:
    """
    relocation_added_costs = []
    all_voyages = sch.flattened_voyages()
    for (origin, target) in combinations(all_voyages, 2):
        for inst in origin.route:
            if is_relocation_feasible(sch, inst, origin, target):
                # OPTIMIZE: remove redundant calculations of removal and insertion costs for same insts and voyages
                relocation_added_costs.append((inst, origin, target, relocation_added_cost(sch, inst, origin, target)))
    if relocation_added_costs is None or len(relocation_added_costs) == 0:
        return []
    return relocation_added_costs


def swap_added_costs(sch):
    """
    Calculates added cost matrix of swapping visits in the schedule sch.
    :param sch:
    :type sch: Schedule
    :return:
    """
    swap_added_costs = []
    for vessel_voyages in sch.schedule.values():
        for i, voyage1 in enumerate(vessel_voyages):
            for j, voyage2 in enumerate(vessel_voyages):
                if i == j:
                    continue
                for inst1 in voyage1.route:
                    for inst2 in voyage2.route:
                        if sch.is_swap_feasible(inst1, voyage1, inst2, voyage2):
                            swap_added_costs.append(
                                (inst1, voyage1, inst2, voyage2, sch.swap_added_cost(inst1, voyage1, inst2, voyage2)))
    swap_added_costs = sorted(swap_added_costs, key=lambda x: x[4])
    return swap_added_costs
