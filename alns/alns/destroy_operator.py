from config.config_utils import get_config
from alns.Beans.schedule import Schedule
import numpy as np
import math
import random
from alns.alns.mutation_service import added_costs_for_visits_removal, update_removal_added_costs

eps_min = float(get_config()['alns.random_removal']['eps_min'])
eps_max = float(get_config()['alns.random_removal']['eps_max'])


def random_removal_vlad(schedule: Schedule, visit_pool: set):
    voyage_list = list(np.concatenate(schedule.schedule.values()).flat)
    number_of_voyages_to_destroy = math.ceil(eps_max*len(voyage_list))
    voyages_to_destroy = random.sample(voyage_list, number_of_voyages_to_destroy)
    for voyage in voyages_to_destroy:
        visit_number = len(voyage.route)
        min_vis_number_to_remove = round(eps_min*visit_number)
        max_vis_number_to_remove = round(eps_max*visit_number)
        vis_number_to_remove = random.randint(min_vis_number_to_remove, max_vis_number_to_remove)
        insts_to_free = random.sample(voyage.route, vis_number_to_remove)
        if visit_number == vis_number_to_remove:
            schedule.schedule[voyage.vessel.name].remove(voyage)
            for inst in insts_to_free:
                visit_pool.add([inst, voyage.start_time])
        else:
            for inst in insts_to_free:
                # voyage.route.remove(inst)
                schedule.remove_visit(voyage, inst)
                visit_pool.add([inst, voyage.start_time])
            voyage.calc_voyage_end_time(voyage.route)


def random_removal(schedule: Schedule):
    removed_insts_pool = []
    voyage_list = list(np.concatenate(list(schedule.schedule.values())).flat)
    number_of_voyages_to_destroy = math.ceil(eps_max*len(voyage_list))
    voyages_to_destroy = random.sample(voyage_list, number_of_voyages_to_destroy)
    random.seed(42)
    for voyage in voyages_to_destroy:
        visit_number = len(voyage.route)
        min_vis_number_to_remove = round(eps_min*visit_number)
        max_vis_number_to_remove = round(eps_max*visit_number)
        vis_number_to_remove = random.randint(min_vis_number_to_remove, max_vis_number_to_remove)
        visits_to_remove = random.sample(voyage.route, vis_number_to_remove)
        schedule.remove_visits_from_voyage(visits_to_remove, voyage)
        for inst in visits_to_remove:
            removed_insts_pool.append(inst)
    return removed_insts_pool


def worst_removal(schedule, n_remove_visits, p=5):
    removed_insts_pool = []
    all_visits = schedule.all_visits()
    visit_added_costs = added_costs_for_visits_removal(visits=all_visits, sch=schedule)
    # print(visit_added_costs)
    random.seed(42)
    while len(removed_insts_pool) < n_remove_visits:
        y = random.uniform(0, 1)
        idx_visit_to_remove = round(pow(y, p) * len(visit_added_costs))
        visit, _ = visit_added_costs[idx_visit_to_remove]
        schedule.remove_visit(visit)
        removed_insts_pool.append(visit.inst)
        visit_added_costs = update_removal_added_costs(visit_added_costs, visit, schedule)
    return removed_insts_pool


def worst_removal_freak(schedule, n_remove_visits, p=5):
    removed_visits_pool = []
    all_visits = schedule.all_visits()
    visit_added_costs = added_costs_for_visits_removal(visits=all_visits, sch=schedule)
    random.seed(42)
    while len(removed_visits_pool) < n_remove_visits:
        y = random.uniform(0, 1)
        idx_visit_to_remove = round(pow(y, p) * len(visit_added_costs))
        visit, _ = visit_added_costs[idx_visit_to_remove]
        schedule.remove_visit(visit)
        removed_visits_pool.append(visit)
        visit_added_costs = update_removal_added_costs(visit_added_costs, visit, schedule)
    return removed_visits_pool


def shaw_removal(schedule, voyage_pool):
    pass
