import numpy as np
from scipy.optimize import differential_evolution
from .base import OptimizationAlgorithm

class GeneticAlgorithm(OptimizationAlgorithm):
    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.bounds = [
            (0.01, 50),    # alpha1
            (0.01, 50),    # alpha2
            (0.01, 50),    # alpha3
            (0.01, 50),    # alpha4
            (0.001, 100),  # delta1
            (0.001, 100),  # delta2
            (0.01, 250),   # Kd
            (1, 5)         # n
        ]

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
        return -evaluation

    def optimize_parameters(self, population_size: int, generations: int):
        result = differential_evolution(
            func=self._evaluate_function,
            bounds=self.bounds,
            popsize=population_size,
            strategy='best1bin',
            maxiter=generations,
            tol=1e-4,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            x0=self.starting_point
        )

        best_params_dict = {
            'alpha1': result.x[0],
            'alpha2': result.x[1],
            'alpha3': result.x[2],
            'alpha4': result.x[3],
            'delta1': result.x[4],
            'delta2': result.x[5],
            'Kd': result.x[6],
            'n': result.x[7]
        }
        return best_params_dict
