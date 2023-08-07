import numpy as np

from alns.Beans.schedule import Schedule
from alns.Beans.visit import Visit


def removal_added_cost(visit, sch):
    """
    Calculates added cost of having visit in the schedule  sch.
    :param visit:
    :type visit: Visit
    :param sch:
    :type sch: Schedule
    :return:
    :rtype: float|bool
    """
    voyage = visit.voyage
    new_voyage = voyage.__deepcopy__()
    new_voyage.route = [i for i in new_voyage.route if i != visit.inst]
    new_voyage.improve_full_enum()
    old_cost = voyage.calc_variable_cost()
    new_cost = new_voyage.calc_variable_cost()
    return new_cost - old_cost


def insertion_added_cost(inst, target_voyage, sch):
    """
    Calculates added cost of inserting installation visit in the schedule sch.
    :param target_voyage:
    :type target_voyage: Voyage
    :param sch:
    :type sch: Schedule
    :param inst:
    :type inst: Installation
    :return:
    :rtype: float|bool
    """
    if inst.idx == 7 and target_voyage.vessel == 1:
        a = 1
    if inst in target_voyage.route:
        return np.inf
    new_voyage = target_voyage.__deepcopy__()
    new_voyage.add_visit(inst)
    new_voyage.improve_full_enum()
    if sch.check_for_replacement_overlap(new_voyage, target_voyage):
        return np.inf
    old_cost = target_voyage.calc_variable_cost()
    new_cost = new_voyage.calc_variable_cost()
    return new_cost - old_cost


def is_departure_spread_ok(sch, inst, target_voyage):
    """
    :param sch:
    :param inst:
    :param target_voyage:
    :return:
    """
    departure_spread_is_ok = True
    for voyage in sch.find_voyages_containing_visit(inst):
        if voyage == target_voyage:
            continue
        if abs(voyage.start_day - target_voyage.start_day) <= inst.departure_spread:
            departure_spread_is_ok = False
            break
    return departure_spread_is_ok


def is_insertion_feasible(inst, target_voyage, sch):
    """
    Checks if insertion of visit in voyage is feasible.
    :param inst:
    :param target_voyage:
    :param sch:
    :return:
    :rtype: bool
    """
    visit_is_duplicated = inst in target_voyage.route
    # TODO: add check for capacity violation?
    departure_spread_is_ok = is_departure_spread_ok(sch, inst, target_voyage)
    return not visit_is_duplicated and departure_spread_is_ok


def _sort_removal_added_costs(visit_added_costs):
    return sorted(visit_added_costs, key=lambda x: x[1], reverse=False)


def added_costs_for_visits_removal(visits, sch):
    added_costs = []
    for visit in visits:
        added_costs.append(removal_added_cost(visit, sch))
    visit_added_costs = list(zip(visits, added_costs))
    visit_added_costs = _sort_removal_added_costs(visit_added_costs)
    return visit_added_costs


def added_costs_for_visits_insertion(insts, sch):
    """
    Calculates added cost matrix of inserting insts as visits into schedule sch.
    The matrix is sorted by added cost ascending for each inst.
    :param insts:
    :type insts: list[Installation]
    :param sch:
    :type sch: Schedule
    :return:
    """
    added_costs = []
    voyages = [voyage for vessel_voyages in sch.schedule.values() for voyage in vessel_voyages]
    for inst in insts:
        visit_added_costs = []
        for voyage in voyages:
            if is_insertion_feasible(inst, voyage, sch):
                iac = insertion_added_cost(inst, voyage, sch)
                visit_added_costs.append((voyage, insertion_added_cost(inst, voyage, sch)))
            else:
                visit_added_costs.append((voyage, np.inf))
        added_costs.append(_sort_removal_added_costs(visit_added_costs))
    return list(zip(insts, added_costs))


def update_removal_added_costs(visit_added_costs, removed_visit, sch):
    """
    Updates the added costs of visits in visit_added_costs after removal of removed_visit.
    :param visit_added_costs:
    :type visit_added_costs: list[tuple(Visit, float)]
    :param removed_visit:
    :type removed_visit: Visit
    :param sch:
    :type sch: Schedule
    :return:
    """

    def same_voyage(x, y):
        return x.voyage == y.voyage

    def same_inst(x, y):
        return x.inst == y.inst

    visit_added_costs_up_to_date = [(visit, acost) for (visit, acost) in visit_added_costs
                                    if not same_voyage(visit, removed_visit)]
    visits_added_cost_to_update = [visit for (visit, acost) in visit_added_costs
                                   if same_voyage(visit, removed_visit) & (not same_inst(visit, removed_visit))]
    updated_visit_added_costs = added_costs_for_visits_removal(visits_added_cost_to_update, sch)
    return _sort_removal_added_costs(visit_added_costs_up_to_date + updated_visit_added_costs)


def update_insertion_added_costs(visit_added_costs, inserted_visit, sch):
    """
    Updates the added costs of visits in visit_added_costs after insertion of inserted_visit.
    :param visit_added_costs:
    :type visit_added_costs: list[tuple(Visit, float)]
    :param inserted_visit:
    :type inserted_visit: Visit
    :param sch:
    :type sch: Schedule
    :return:
    """

    def same_voyage(x, y):
        return x.voyage == y.voyage

    def same_inst(x, y):
        return x.inst == y.inst

    visit_added_costs_up_to_date = [(visit, acost) for (visit, acost) in visit_added_costs
                                    if not same_voyage(visit, inserted_visit)]
    visits_added_cost_to_update = [visit for (visit, acost) in visit_added_costs
                                   if same_voyage(visit, inserted_visit) & (not same_inst(visit, inserted_visit))]
    updated_visit_added_costs = added_costs_for_visits_insertion(visits_added_cost_to_update, sch)
    return _sort_removal_added_costs(visit_added_costs_up_to_date + updated_visit_added_costs)


def min_insertion_added_cost_index(visit_added_costs):
    """
    :param visit_added_costs:
    :type visit_added_costs: list[tuple(Visit, list[tuple(Voyage, float)])]
    :return:
    """
    min_added_cost = np.inf
    min_added_cost_index = None
    for i, (inst, voyage_added_costs) in enumerate(visit_added_costs):
        for voyage, added_cost in voyage_added_costs:
            if added_cost < min_added_cost:
                min_added_cost = added_cost
                min_added_cost_index = i
    return min_added_cost_index
