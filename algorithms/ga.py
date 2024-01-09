import numpy as np
from pyDOE import lhs
from .base import OptimizationAlgorithm

class GeneticAlgorithm(OptimizationAlgorithm):
    def __init__(self, evaluator, mutation_rate=0.35, elite_percentage=0.2, **kwargs):
        super().__init__(evaluator, **kwargs)
        self.param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
        self.mutation_rate = mutation_rate
        self.elite_percentage = elite_percentage

    def initialize_population(self):
        lhd = lhs(len(self.bounds), samples=self.population_size, criterion='center')
        scaled_samples = lhd * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        return scaled_samples

    def _mutate(self, individuals):
        mask = np.random.rand(*individuals.shape) < self.mutation_rate
        mutation_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        mutation = np.random.uniform(low=-mutation_range, high=mutation_range, size=individuals.shape)
        individuals += mask * mutation
        np.clip(individuals, self.bounds[:, 0], self.bounds[:, 1], out=individuals)
        return individuals

    def _evaluate_function(self, params, **kwargs):
        self.evaluator.set_params_list(params)
        return self.evaluator.evaluate(weights=self.weigths, joined=self.joined, **kwargs)

    def evolve(self, base_population):
        population = self.initialize_population() if base_population is None else base_population
        elite_count = int(self.elite_percentage * len(population))
        prev_best, current_best = None, None

        for itt in range(self.generations):
            sorted_indices = np.argsort([-self._evaluate_function(individual) for individual in population])
            elites = population[sorted_indices[:elite_count]]

            num_children = self.population_size - elite_count
            parents_indices = np.random.choice(len(population), size=(num_children, 2), replace=True)
            parent1, parent2 = population[parents_indices[:, 0]], population[parents_indices[:, 1]]
            mask = np.random.rand(num_children, len(self.param_names)) < 0.5
            children = np.where(mask, parent1, parent2)
            children = self._mutate(children)

            population = np.vstack((elites, children))

            best_individual = max(population, key=lambda x: self._evaluate_function(x, export_index=itt, export=self.export_data))
            current_eval = self._evaluate_function(best_individual)
            if prev_best is None or current_eval > prev_best:
                prev_best = current_eval
                self.do_print(f'[{itt}] Update best with {current_eval:.5f}')

            current_best = best_individual

            self.generations_evaluations[itt] = current_eval

        return current_best

    def optimize_parameters(self, population_size: int, generations: int, base_population=None, top=None):
        self.population_size = population_size
        self.generations = generations
        self.create_export_matrix(generations)
        best_solution = self.evolve(base_population)
        self.do_export(top)
        return dict(zip(self.param_names, best_solution))