from alns.Beans.schedule import Schedule
from alns.utils.distance_manager import DistanceManager

from alns.alns.destroy_operator import worst_removal, random_removal, shaw_removal
from alns.alns.repair_operator import deep_greedy_insertion, k_regret_insertion
from alns.alns.improve_operator import deep_greedy_swap, deep_greedy_relocation, daly_departure_relocaion,\
    number_of_voyages_reduction, voyage_improvement, fleet_size_and_cost_reduction


class ALNS:
    def __init__(self, installations, base, fleet,
                 iterations: int, speed_coeff: float, operator_select_type="Stochastic") -> None:
        self.insts = installations
        self.base = base
        self.vessels = fleet
        self.distance_manager = DistanceManager(self.base,self.insts)
        self.iterations_num = iterations
        self.speed_up_coeff = speed_coeff
        self.select_operator = operator_select_type

    def start(self, repetitions):
        best_solution = None
        for r in range(repetitions):
            free_visits_pool = set()
            # init_solution = Schedule(self.vessels, self.insts, self.base,self.distance_manager)
            init_solution = Schedule(self.vessels, self.insts, self.base)
            best_solution = init_solution
            current_solution = init_solution
            for i in range(self.iterations_num):
                operators = self.get_operators_combination(current_solution)
                broken_solution = operators[0](current_solution, free_visits_pool)
                repaired_solution = operators[1](broken_solution, free_visits_pool)
                if self.check_feasibility(repaired_solution) and not free_visits_pool:
                    is_improved = True
                    improved_solution = repaired_solution
                    while is_improved:
                        improved_solution, is_improved = operators[2](repaired_solution)

                    if improved_solution.solution_cost < best_solution.solution_cost:
                        best_solution = improved_solution
                        current_solution = improved_solution
                    elif i < self.speed_up_coeff * self.iterations_num:
                        if improved_solution.solution_cost < current_solution.solution_cost or \
                                self.accept(improved_solution, current_solution):
                            current_solution = improved_solution
                    else:
                        current_solution = best_solution
        return best_solution

    def get_operators_combination(self, solution):
        if self.select_operator == "Stochastic":
            return [random_removal, deep_greedy_insertion, worst_removal]
        elif self.select_operator == "RL":
            return []

    def check_feasibility(self, solution):
        pass

    def accept(self, new_solution, prev_solution):
        return False
