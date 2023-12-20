import numpy as np
from .base import OptimizationAlgorithm

class AntColonyOptimization(OptimizationAlgorithm):
    def __init__(self, evaluator, num_ants=20, max_iterations=200, alpha=2.5, rho=0.5, elite=0.2, joined=[]):
        super().__init__(evaluator, joined=joined)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.rho = rho
        self.bounds = [
            (0.01, 50),
            (0.01, 50),
            (0.01, 50),
            (0.01, 50),
            (0.001, 100),
            (0.001, 100),
            (0.01, 250),
            (1, 5)
        ]
        self.param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
        self.num_params = len(self.bounds)
        self.elite_percentage = elite
        self.num_elite = int(self.num_ants * self.elite_percentage)
        self.starting_point = np.array(self.starting_point)

    def _initialize_pheromones(self):
        self.pheromones = np.ones(self.num_params)

    def _move_ant(self):
        if np.random.rand() < 0.2:
            candidate_solution = [np.random.uniform(low=low, high=high) * 10 for low, high in self.bounds]
        else:
            candidate_solution = self.starting_point + np.random.uniform(-self.alpha, self.alpha, self.num_params)
            candidate_solution = np.clip(candidate_solution, [low for low, _ in self.bounds], [high for _, high in self.bounds])

        return candidate_solution

    def _update_pheromones(self, solutions, scores):
        self.pheromones *= (1 - self.rho)
        self.pheromones += np.dot(solutions.T, scores)

    def _evaluate_function(self, params):
        params_dict = dict(zip(self.param_names, params))
        return self.evaluator(*self.joined, **params_dict).evaluate()

    def optimize_parameters(self):
        self._initialize_pheromones()

        best_solution = None
        best_score = -np.inf

        for itt in range(self.max_iterations):
            ant_solutions = np.array([self._move_ant() for _ in range(self.num_ants)])
            ant_scores = np.array([self._evaluate_function(params) for params in ant_solutions])

            if np.max(ant_scores) > best_score:
                best_solution = ant_solutions[np.argmax(ant_scores)]
                best_score = np.max(ant_scores)
                print(f'[{itt}] Update best with {best_score:.5f}')

            elite_indices = np.argsort(ant_scores)[-self.num_elite:]
            elite_solutions = ant_solutions[elite_indices]
            elite_scores = ant_scores[elite_indices]

            self._update_pheromones(ant_solutions, ant_scores)
            self._update_pheromones(elite_solutions, elite_scores)

        return dict(zip(self.param_names, best_solution))
