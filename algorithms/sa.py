import numpy as np
from .base import OptimizationAlgorithm

class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self, evaluator):
        super().__init__(evaluator)

        self.current_solution = self.starting_point
        self.best_solution = self.starting_point
        self.temperature = 100.0  # Initial temperature
        self.min_temperature = 1e-6  # Minimum temperature
        self.alpha = 0.9  # Cooling rate

    def _evaluate_function(self, params):
        params_dict = {
            'alpha1': params[0],
            'alpha2': params[1],
            'alpha3': params[2],
            'alpha4': params[3],
            'delta1': params[4],
            'delta2': params[5],
            'Kd': params[6],
            'n': params[7]
        }
        evaluation = self.evaluator(**params_dict).evaluate()
        return evaluation

    def _get_neighbor(self):
        neighbor = self.current_solution + np.random.normal(0, 1, len(self.current_solution)) * self.temperature
        return neighbor.tolist()

    def optimize_parameters(self):
        while self.temperature > self.min_temperature:
            new_solution = self._get_neighbor()
            new_solution_score = self._evaluate_function(new_solution)
            current_score = self._evaluate_function(self.current_solution)

            if new_solution_score > current_score or np.exp((new_solution_score - current_score) / self.temperature) > np.random.random():
                self.current_solution = new_solution

            if new_solution_score > self._evaluate_function(self.best_solution):
                self.best_solution = new_solution

            self.temperature *= self.alpha

        best_params_dict = {
            'alpha1': self.best_solution[0],
            'alpha2': self.best_solution[1],
            'alpha3': self.best_solution[2],
            'alpha4': self.best_solution[3],
            'delta1': self.best_solution[4],
            'delta2': self.best_solution[5],
            'Kd': self.best_solution[6],
            'n': self.best_solution[7]
        }

        return best_params_dict
