import numpy as np
from .base import OptimizationAlgorithm

class AntColonyOptimization(OptimizationAlgorithm):
    def __init__(self, evaluator, num_ants=20, max_iterations=200, alpha=10.5, rho=0.5, elite=0.2, **kwargs):
        super().__init__(evaluator, **kwargs)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.rho = rho
        self.param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
        self.num_params = len(self.bounds)
        self.elite_percentage = elite
        self.num_elite = int(self.num_ants * self.elite_percentage)
        self.bounds = np.array(self.bounds)

    def _initialize_pheromones(self):
        self.pheromones = np.ones(self.num_params)

    def _move_ants(self):
        return np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1],
                                 size=(self.num_ants, 8))

    def _update_pheromones(self, solutions, scores):
        self.pheromones *= (1 - self.rho)
        self.pheromones += np.dot(solutions.T, scores)

    def _evaluate_function(self, params, **kwargs):
        self.evaluator.set_params_list(params)
        return self.evaluator.evaluate(weights=self.weigths, joined=self.joined, **kwargs)

    def optimize_parameters(self, population_size: int, generations: int, top=None):
        self.num_ants = population_size
        self.max_iterations = generations
        self._initialize_pheromones()
        self.create_export_matrix(generations)

        best_solution = None
        best_score = -np.inf

        for itt in range(self.max_iterations):
            ant_solutions = self._move_ants()
            ant_scores = np.array([self._evaluate_function(params, export_index=itt, export=self.export_data) for params in ant_solutions])

            if np.max(ant_scores) > best_score:
                best_solution = ant_solutions[np.argmax(ant_scores)]
                best_score = np.max(ant_scores)
                self.do_print(f'[{itt}] Update best with {best_score:.5f}')

            self.generations_evaluations[itt] = best_solution

            elite_indices = np.argsort(ant_scores)[-self.num_elite:]
            elite_solutions = ant_solutions[elite_indices]
            elite_scores = ant_scores[elite_indices]

            self._update_pheromones(ant_solutions, ant_scores)
            self._update_pheromones(elite_solutions, elite_scores)

        self.do_export(top)
        return dict(zip(self.param_names, best_solution))
