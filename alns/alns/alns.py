from alns.Beans.schedule import Schedule
from alns.utils.utils import *
from alns.alns.destroy_operator import *
from alns.alns.repair_operator import *
from alns.alns.improve_operator import *
from config.config_utils import get_config
import logging
from datetime import datetime
from alns.utils.utils import format_td


def calculate_operator_probabilities(weights):
    return [weight / sum(weights) for weight in weights]




def select_operator(operators, probabilities):
    """
    Selects an operator based on the given probabilities. Returns the operator and its index in the list of operators.
    :param operators:
    :param probabilities:
    :return:
    """
    selected_operator = np.random.choice(operators, p=probabilities)
    logging.debug(f"Selected operator: {selected_operator.__name__}")
    return selected_operator, operators.index(selected_operator)


class ALNS:
    def __init__(self, installations, base, vessels,
                 operator_selection_type="Standard") -> None:
        self.installations = installations
        self.base = base
        self.vessels = vessels
        self.num_restarts = int(get_config()['alns']['num_restarts'])
        self.num_iterations = int(get_config()['alns']['num_iterations'])
        self.aggressive_search_factor = float(get_config()['alns']['aggressive_search_factor'])
        self.num_iterations_weights_update = int(get_config()['alns']['num_iterations_weights_update'])
        self.reaction_factor = float(get_config()['alns']['reaction_factor'])
        self.cooling_parameter = float(get_config()['alns']['cooling_parameter'])
        self.cooling_rate = 1 - self.cooling_parameter / self.num_iterations
        self.operator_selection_type = operator_selection_type
        self._temperature = None
        self._init_logging()
        self._init_algorithm()

    def _init_logging(self):
        logging.basicConfig(
            filename=f"logs/ALNS_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
            level=logging.DEBUG,  # Set the desired logging level (e.g., DEBUG, INFO, WARNING, ERROR)
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Create a logger for your module
        self.logger = logging.getLogger(__name__)

    def _init_algorithm(self):
        if self.operator_selection_type == 'Standard':
            self.destroy_operators = [worst_removal, random_removal]
            self.repair_operators = [deep_greedy_insertion, k_regret_insertion]
            self.improve_operators = [number_of_voyages_reduction, fleet_size_and_cost_reduction,
                                      deep_greedy_relocation, deep_greedy_swap]
            self.improve_operators_sequence = [
                #number_of_voyages_reduction,
                #fleet_size_and_cost_reduction,
                deep_greedy_relocation,
                fleet_size_and_cost_reduction,
                deep_greedy_swap_plain,
                fleet_size_and_cost_reduction
            ]
            self.destroy_operator_weights = [1] * len(self.destroy_operators)
            self.repair_operator_weights = [1] * len(self.repair_operators)
            self.destroy_operator_probabilities = calculate_operator_probabilities(self.destroy_operator_weights)
            self.repair_operator_probabilities = calculate_operator_probabilities(self.repair_operator_weights)
            self.rewards = [1, 0.5, 0]


    def run(self):
        self.logger.info("Starting ALNS run")
        start_time = datetime.now()
        best_cost = np.inf
        best_solution = None
        for i in range(self.num_restarts):
            self.logger.info(f"Restart {i+1}/{self.num_restarts}")
            s_0 = self.initial_solution()
            s_star = s_0.shallow_copy()
            s = s_0.shallow_copy()
            _temperature_init = s_0.total_cost
            self._temperature = _temperature_init
            for j in range(self.num_iterations):
                iteration_start_time = datetime.now()
                self._temperature = self._temperature * self.cooling_rate
                s_1 = s.shallow_copy()
                iteration_start_cost = s_1.total_cost
                self.logger.debug(f"Iteration {j+1} starting")
                restoration_start_time = datetime.now()
                destroy_operator, d_id = select_operator(self.destroy_operators, self.destroy_operator_probabilities)
                repair_operator, r_id = select_operator(self.repair_operators, self.repair_operator_probabilities)
                inst_pool = destroy_operator(s_1)
                s_1.insert_idle_vessel_and_add_empty_voyages()
                pool_empty = repair_operator(inst_pool, s_1)
                s_1.drop_empty_voyages()
                s_1.update()
                operators_cumm_time = timedelta(0)
                operators_cumm_time += datetime.now() - restoration_start_time
                self.logger.debug(f"[Restoration] time:{format_td(datetime.now() - restoration_start_time)}, "
                                  f"stage cost change: {s_1.total_cost - iteration_start_cost:0.2f}, "
                                  f"pool empty: {pool_empty}, feasible: {s_1.feasible}")
                improve_start_cost = s_1.total_cost
                if pool_empty and s_1.feasible:
                    is_improved = True
                    while is_improved:
                        for improve_operator in self.improve_operators_sequence:
                            operator_start_cost = s_1.total_cost
                            improve_operator_start_time = datetime.now()
                            s_1 = improve_operator(s_1)
                            operators_cumm_time += datetime.now() - improve_operator_start_time
                            self.logger.debug(f"[Improve] {improve_operator.__name__:30} "
                                              f"time:{format_td(datetime.now() - improve_operator_start_time)}, "
                                              f"cost change: {s_1.total_cost - operator_start_cost:>6.2f}")
                        else:
                            is_improved = False
                    self.logger.debug(f"[Improve] stage cost change: {s_1.total_cost - improve_start_cost:0.2f}")

                    if j < self.aggressive_search_factor * self.num_iterations:
                        if s_1.total_cost < s_star.total_cost:
                            s_star = s_1.shallow_copy()
                            s = s_1
                            self.logger.debug(f"[Best] found cost: {s_star.total_cost:0.2f}")
                        elif s_1.total_cost < s.total_cost:
                            s = s_1
                            self.logger.debug(f"[New] cost: {s.total_cost:0.2f}")
                        elif self.accept(s, s_1):
                            s = s_1
                            self.logger.debug(f"[Annealing] cost: {s.total_cost:0.2f}")
                    elif s_1.total_cost < s_star.total_cost:
                        s_star = s_1.shallow_copy()
                        s = s_1
                        self.logger.debug(f"[Best] found cost: {s_star.total_cost:0.2f}")
                    else:
                        s = s_star
                        self.logger.debug(f"Continue with best solution: {s_star.total_cost:0.2f}")
                    iteration_end_time = datetime.now()
                    self.logger.debug(f"Iteration {j+1} ends, total_time: {format_td(iteration_end_time-iteration_start_time)}," +
                                      f"added time: {format_td(iteration_end_time - iteration_start_time - operators_cumm_time)}")
            self.logger.info('*'*50)
            self.logger.info('Restart completed, best solution found:')
            self.logger.info(f"Total cost: {s_star.total_cost:0.2f}")
            if s_star.total_cost < best_cost:
                best_cost = s_star.total_cost
                best_solution = s_star.shallow_copy()
        end_time = datetime.now()
        self.logger.info(f"ALNS run completed in {format_td(end_time - start_time)}")
        self.logger.info(f"Best solution found: {best_cost:0.2f}")
        return best_solution

    def initial_solution(self):
        sch = Schedule(self.installations, self.vessels, self.base)
        return sch

    def accept(self, s, s_1):
        """
        Simulated annealing acceptance criterion.
        :param s: schedule s
        :type s: Schedule
        :param s_1: schedule s_1
        :type s_1: Schedule
        :return:
        """
        self.logger.debug(f"Acceptance criterion: {np.exp((s.total_cost - s_1.total_cost) / self._temperature):0.2f}")
        return np.random.uniform(0, 1) < np.exp((s.total_cost - s_1.total_cost) / self._temperature)