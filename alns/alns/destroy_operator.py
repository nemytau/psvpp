from config.config_utils import get_config
import numpy as np
import math
import random

eps_min = float(get_config()['alns.random_removal']['eps_min'])
eps_max = float(get_config()['alns.random_removal']['eps_max'])


def random_removal(schedule: dict, visit_pool: set):
    voyage_list = list(np.concatenate(schedule.values()).flat)
    number_of_voyages_to_destroy = math.ceil(eps_max*len(voyage_list))
    voyages_to_destroy = random.sample(voyage_list,number_of_voyages_to_destroy)
    for voyage in voyages_to_destroy:
        visit_number = len(voyage.route)
        min_vis_number_to_remove = round(eps_min*visit_number)
        max_vis_number_to_remove = round(eps_max*visit_number)
        vis_number_to_remove = random.randint(min_vis_number_to_remove, max_vis_number_to_remove)
        insts_to_free = random.sample(voyage.route,vis_number_to_remove)
        if visit_number == vis_number_to_remove:
            schedule[voyage.vessel.name].remove(voyage)
            for inst in insts_to_free:
                visit_pool.add([inst, voyage.start_time])
        else:
            for inst in insts_to_free:
                voyage.route.remove(inst)
                visit_pool.add([inst, voyage.start_time])
            voyage.calc_voyage_length(voyage.route)






def worst_removal(schedule, voyage_pool):
    pass


def Shaw_removal(schedule, voyage_pool):
    pass
