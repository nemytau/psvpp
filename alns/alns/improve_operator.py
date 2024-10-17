from alns.alns.repair_operator import *
import numpy as np
from itertools import permutations, combinations
from alns.alns.destroy_operator import *
from alns.Beans.schedule import Schedule
import random

DAYS = 7


def fleet_size_reduction(schedule):
    """
    Reduces the fleet size and cost by removing vessels that are not used.
    :param schedule:
    :type schedule: Schedule
    :return:
    """
    # schedule -> curr_schedule
    curr_schedule = schedule.shallow_copy()
    is_improved = True
    count = 0
    while is_improved:
        count += 1
        vessels = curr_schedule.get_nonempty_vessels()
        max_voyages = max([len(voyages) for voyages in curr_schedule.schedule.values()])
        l = np.full(shape=(len(vessels), max_voyages, len(vessels)), fill_value=np.nan, dtype=np.float32)
        vessel_pairs_indices = list(permutations(range(len(vessels)), 2))
        for origin_v_idx, target_v_idx in vessel_pairs_indices:
            origin_v, target_v = vessels[origin_v_idx], vessels[target_v_idx]
            for i, voyage_to_move in enumerate(curr_schedule.schedule[origin_v]):
                if voyage_to_move.check_load_feasibility(target_v):
                    l[origin_v_idx, i, target_v_idx] = curr_schedule.calculate_reassignment_overlap(voyage_to_move,
                                                                                                    target_v)
                else:
                    l[origin_v_idx, i, target_v_idx] = np.inf
        l_least_for_target_v = np.nanmin(l, axis=2)
        target_v_idx_for_least_l = np.nanargmin(np.nan_to_num(l, nan=np.inf), axis=2)
        least_overlap = np.nanmin(np.nansum(l_least_for_target_v, axis=1))
        least_overlap_vessel_idx = np.argmin(np.nansum(l_least_for_target_v, axis=1))
        # TODO: replace np.inf with some value (find experimentally)
        max_least_overlap = np.inf
        if least_overlap < max_least_overlap:
            sch = curr_schedule.shallow_copy()
            removed_insts_pool = []
            for i, voyage_to_move in enumerate(curr_schedule.schedule[vessels[least_overlap_vessel_idx]]):
                target_vessel_idx = target_v_idx_for_least_l[least_overlap_vessel_idx, i]
                actual_voyage_to_move = voyage_to_move
                while sch.calculate_reassignment_overlap(actual_voyage_to_move, vessels[target_vessel_idx]) > 0:
                    voyages_to_shrink = sch.voyages_to_shrink_to_fit(actual_voyage_to_move, vessels[target_vessel_idx])
                    # NOTE: in the paper it is done for each voyage separately, so it removes at least one inst from each voyage
                    # Here we remove only one inst from the worst voyage at the time
                    removed_insts_pool.extend(worst_removal(sch, 1, voyages=voyages_to_shrink))
                    actual_voyage_to_move = sch.find_voyage_on_day(voyage_to_move.vessel, voyage_to_move.start_day)
                    if actual_voyage_to_move is None:
                        break
                if actual_voyage_to_move is not None:
                    sch.reassign_voyage_to_vessel(actual_voyage_to_move, vessels[target_vessel_idx])
            # NOTE 1: in the paper it is done for each voyage, so it adds same number of empty voyages as removed
            # It increases the number of possible insertions for 2-regret insertion, which are pretty the same
            # NOTE 2: in the paper it is done for each voyage, so it may modify voyages that are planned to move.
            # So precalculated l matrix is not valid anymore. It may cause some problems with reassignment.
            # Also, it potentially increases number of removals. So, I am doing it only once.
            # To do it for each voyage, Tab next 4 lines.
            # And move init of removed_insts_pool to the beginning of the loop
            sch.insert_idle_vessel_and_add_empty_voyages()
            k_regret_insertion(removed_insts_pool, sch)
            sch.drop_empty_voyages()
            sch.update()
            if sch.feasible:
                sch.drop_empty_voyages()
                is_improved = sch.total_cost < curr_schedule.total_cost
                curr_schedule = sch
            else:
                is_improved = False
        if count > 50:
            print("Too many iterations Fleet Size Reduction")
            assert False
    return curr_schedule


def cost_reduction(schedule):
    is_improved = True
    curr_schedule = schedule.shallow_copy()
    while is_improved:
        sch = curr_schedule.shallow_copy()
        for origin_v, voyages in sch.schedule.items():
            if len(voyages) != 0:
                cheapest_v_for_relocation = None
                for target_v in sch.vessels:
                    can_be_reassigned = True
                    if target_v.cost >= origin_v.cost:
                        continue
                    if cheapest_v_for_relocation is not None and target_v.cost >= cheapest_v_for_relocation.cost:
                        continue
                    for voyage in voyages:
                        if (not voyage.check_load_feasibility(target_v)) or \
                                sch.check_for_insertion_overlap(voyage, target_v):
                            can_be_reassigned = False
                            break
                    if can_be_reassigned:
                        cheapest_v_for_relocation = target_v
                        break
                if cheapest_v_for_relocation is not None:
                    sch.force_reassign_voyages(origin_v, cheapest_v_for_relocation)
        sch.update()
        if sch.feasible:
            is_improved = sch.total_cost < curr_schedule.total_cost
            curr_schedule = sch
            break
    return curr_schedule


def fleet_size_and_cost_reduction(schedule):
    sch = fleet_size_reduction(schedule)
    sch = cost_reduction(sch)
    return sch

def number_of_voyages_reduction(schedule):
    """
    Reduces the number of voyages in schedule by removing voyages one by one.
    :param schedule: Schedule to be reduced.
    :type schedule: Schedule.
    :return:
    """
    number_of_voyages_is_reduced = True
    number_reduced = 0
    curr_schedule = schedule.shallow_copy()
    while number_of_voyages_is_reduced:
        sch = curr_schedule.shallow_copy()
        voyages = [voyage.__deepcopy__() for voyage in sch.flattened_voyages()]
        cost_diffs = []
        for voyage in voyages:
            init_cost = sch.calc_total_cost()
            sch.remove_voyage(voyage)
            if deep_greedy_insertion(voyage.route, sch):
                cost_diffs.append(init_cost - sch.calc_total_cost())
            else:
                cost_diffs.append(-np.inf)
            sch = curr_schedule.shallow_copy()
        min_idx = np.argmax(cost_diffs)
        if cost_diffs[min_idx] == -np.inf:
            number_of_voyages_is_reduced = False
        else:
            # TODO: don't update schedule to return sch?
            curr_schedule.remove_voyage(voyages[min_idx])
            all_inserted = deep_greedy_insertion(voyages[min_idx].route, curr_schedule)
            if not all_inserted:
                print("Not all inserted")
            number_reduced += 1
    curr_schedule.update()
    # TODO: return sch?
    return curr_schedule


def deep_greedy_relocation(schedule):
    """
    Relocates voyages from one vessel to another one by one.
    :param schedule: Schedule to be relocated.
    :type schedule: Schedule.
    :return:
    """
    # Почему-то у меня релокейшн делался в случайном порядке, а не лучший вариант
    # TODO: Why is it here?
    sch = schedule.shallow_copy()
    curr_schedule = schedule.shallow_copy()

    # TODO: Remove comments in this block or remove comments at allы???
    # 06.09.2024 I guess by removing comments it was meant to remove them with contains
    # I decided that these relocation added costs are not calculated correctly, as in the calculation of insertion
    # added costs some options are elimenated by the spread requirement. So, I decided to recalculate them from scratch
    # These new calculations are not optimal, but not critical.
    # all_visits = curr_schedule.visited_inst_voyage_list()
    # all_insts = curr_schedule.installations
    # insertion_added_costs = added_costs_for_visits_insertion(all_insts, curr_schedule)
    # removal_added_costs = added_costs_for_visits_removal(all_visits, curr_schedule)
    # relocation_added_costs = relocation_added_costs_from_insertion_and_removal(insertion_added_costs,
    #                                                                            removal_added_costs)

    relocation_added_costs = added_costs_for_visits_relocation(curr_schedule)

    is_improved = True
    # Теперь в цикле сначала делается релокейшн. Если улучшилось и физибл, то обновляется расписание и обновляются
    # added costs. Нужно дописать обновление added costs для релокейшна: обновить только те, которые были затронуты
    # релокейшном.
    # 21.02.2024: Не уверен, что это будет правильно. Что происходит с added costs при релокейшне?
    # - insertion_added_costs: изменяются все косты для to_voyage, могут измениться косты для вояджей по соседству с
    # from_voyage, косты по соседству с to_voyage становятся inf.
    # - removal_added_costs: изменяются все косты для from_voyage и to_voyage, новый кост для to_voyage.
    # В инсершене меняется много, соответсвенно проще заново рассчитать, чем пытаться изменить только затронутые.
    # В ремувале меняется меньше, но не хочу заморачиваться.
    # TODO: Проверить, работает ли релокейшн в рамках одного дня. Возможно, часть опций блочится из-за спреда.
    # 22.02.2024: Вроде все ок по логике, но нужно добавить проверку в коде после вызова этого метода на подсчет
    # количества перемещений внутри одного дня.
    count = 0
    while is_improved:
        # If no relocation possible, it cannot be improved
        # I guess it is duplicating logic of added cost == inf. I think this way is faster than adding dummy added cost
        # with inf cost everytime.
        # I first put it to the init of is_improved place, but error occured in this cyle. So, I moved it here.
        if len(relocation_added_costs) < 1:
            is_improved = False
            break
        count += 1
        sch = curr_schedule.shallow_copy()
        # Best relocation
        inst_to_move, from_voyage, to_voyage, acost = relocation_added_costs[0]
        # NOTE: It happens quite often, hands off.
        if acost == np.inf:
            is_improved = False
            break
        elif acost == -np.inf:
            # TODO: Add to log
            print("Cost is -inf")
            assert False
        sch.relocate_visit(inst_to_move, from_voyage, to_voyage)
        sch.update()
        if not sch.feasible:
            is_improved = False
            # TODO: Add to log
            print("Relocation is not feasible")
        elif sch.total_cost < curr_schedule.total_cost:
            curr_schedule = sch
            # all_visits = curr_schedule.visited_inst_voyage_list()
            # all_insts = curr_schedule.installations
            # insertion_added_costs = added_costs_for_visits_insertion(all_insts, curr_schedule)
            # removal_added_costs = added_costs_for_visits_removal(all_visits, curr_schedule)
            # relocation_added_costs = relocation_added_costs_from_insertion_and_removal(insertion_added_costs,
            #                                                                            removal_added_costs)
            relocation_added_costs = added_costs_for_visits_relocation(curr_schedule)
        else:
            is_improved = False
    # NOTE: for some reason it was in the if below
    curr_schedule.update()
    if curr_schedule.feasible:
        return curr_schedule
    else:
        return schedule


def deep_greedy_swap(schedule):
    is_improved = True
    curr_schedule = schedule.shallow_copy()  # Make an initial shallow copy of the schedule

    while is_improved:
        best_swap = None
        best_swap_cost_reduction = 0  # Track the best possible reduction in cost
        all_visits = curr_schedule.visited_inst_voyage_list()  # Get all visits
        all_swaps = combinations(all_visits, 2)  # Generate all pairwise combinations of visits

        # IMPROVE: Consider parallelizing the swap evaluations to speed up large-scale computations
        # Iterate through all possible swaps
        for swap in all_swaps:
            # Skip invalid swaps (same instance or same voyage)
            if swap[0][0].idx == swap[1][0].idx or swap[0][1] == swap[1][1]:
                continue

            # Perform a shallow copy of the current schedule before testing the swap
            sch = curr_schedule.shallow_copy()

            # Perform the swap and update the schedule
            sch.swap_visits(swap[0], swap[1])
            sch.update()

            # Check if the swap produces a feasible solution and reduces cost
            if sch.feasible:
                cost_reduction = sch.total_cost - curr_schedule.total_cost

                # If the swap provides the best cost reduction so far, store it
                if cost_reduction < best_swap_cost_reduction and cost_reduction < 0:
                    best_swap = swap
                    best_swap_cost_reduction = cost_reduction

        # If a valid best swap was found, apply it to the current schedule
        if best_swap is not None:
            curr_schedule.swap_visits(best_swap[0], best_swap[1])  # Apply the best swap to the current schedule
            curr_schedule.update()  # Update the current schedule after applying the swap
        else:
            is_improved = False  # No further improvements possible, exit the loop

    curr_schedule.update()  # Final update after all swaps are done

    # Return the improved schedule if feasible, otherwise return the original schedule
    return curr_schedule if curr_schedule.feasible else schedule


def deep_greedy_swap_plain(schedule):
    is_improved = True
    curr_schedule = schedule.shallow_copy()  # Make an initial shallow copy of the schedule

    while is_improved:
        best_swap = None
        best_swap_cost_reduction = 0  # Track the best possible reduction in cost
        all_visits = curr_schedule.visits_list_plain()  # Get all visits in plain representation
        all_swaps = combinations(all_visits, 2)  # Generate all pairwise combinations of visits

        # IMPROVE: Consider parallelizing swap evaluations for large visit sets.
        for swap in all_swaps:
            # swap is a pair of tuples (inst, vessel, start_day)
            # Skip invalid swaps (same instance, or same vessel and start day)
            if swap[0][0].idx == swap[1][0].idx or (swap[0][1] == swap[1][1] and swap[0][2] == swap[1][2]):
                continue

            # Perform a shallow copy of the current schedule before testing the swap
            sch = curr_schedule.shallow_copy()

            # Perform the swap using tuple representation and update the schedule
            sch.swap_visits_tuple_repr(swap[0], swap[1])
            sch.update()

            # Check if the swap produces a feasible solution and reduces cost
            if sch.feasible:
                cost_reduction = sch.total_cost - curr_schedule.total_cost

                # If the swap provides the best cost reduction so far, store it
                if cost_reduction < best_swap_cost_reduction and cost_reduction < 0:
                    best_swap = swap
                    best_swap_cost_reduction = cost_reduction

        # If a valid best swap was found, apply it to the current schedule
        if best_swap is not None:
            curr_schedule.swap_visits_tuple_repr(best_swap[0], best_swap[1])  # Apply the best swap
            curr_schedule.update()  # Update the current schedule after applying the swap
        else:
            is_improved = False  # No further improvements possible, exit the loop

    curr_schedule.update()  # Final update after all swaps are done

    # IMPROVE: Explore heuristic-based pruning of the swap space to reduce time complexity.
    # Return the improved schedule if feasible, otherwise return the original schedule
    return curr_schedule if curr_schedule.feasible else schedule

def voyage_improvement():
    pass
