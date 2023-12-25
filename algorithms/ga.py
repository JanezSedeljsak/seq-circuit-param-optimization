import numpy as np
from pyDOE import lhs
from .base import OptimizationAlgorithm

class GeneticAlgorithm(OptimizationAlgorithm):
    def __init__(self, evaluator, mutation_rate=0.35, elite_percentage=0.2, **kwargs):
        super().__init__(evaluator, **kwargs)
        self.param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
        self.mutation_rate = mutation_rate
        self.elite_percentage = elite_percentage
        self.bounds = np.array(self.bounds)

    def initialize_population(self):
        num_samples = self.population_size * 20  # Make the base population 100 times larger
        lhd = lhs(len(self.bounds), samples=num_samples, criterion='center')
        scaled_samples = np.zeros_like(lhd)

        for i in range(len(self.bounds)):
            scaled_samples[:, i] = lhd[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]

        fitness = np.array([-self._evaluate_function(params) for params in scaled_samples])
        sorted_indices = np.argsort(fitness)
        return scaled_samples[sorted_indices[:self.population_size]]

    def _mutate(self, individual):
        mask = np.random.rand(*individual.shape) < self.mutation_rate
        mutation_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        mutation = np.random.uniform(low=-mutation_range, high=mutation_range, size=len(self.param_names))
        individual += mask * mutation
        individual = np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])
        return individual

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

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parents_indices = np.random.choice(len(population), size=2, replace=False)
                parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]
                child = np.where(np.random.rand(len(self.param_names)) < 0.5, parent1, parent2)
                child = self._mutate(child)
                new_population = np.vstack((new_population, child))

            population = new_population

            best_individual = max(population, key=lambda x: self._evaluate_function(x, export_index=itt, export=self.export_data))
            current_eval = self._evaluate_function(best_individual)
            if prev_best is None or current_eval > prev_best:
                prev_best = current_eval
                print(f'[{itt}] Update best with {current_eval:.5f}')

            current_best = best_individual

        return current_best

    def optimize_parameters(self, population_size: int, generations: int, base_population=None):
        self.population_size = population_size
        self.generations = generations
        self.create_export_matrix(generations)
        best_solution = self.evolve(base_population)
        self.do_export()
        return dict(zip(self.param_names, best_solution))
