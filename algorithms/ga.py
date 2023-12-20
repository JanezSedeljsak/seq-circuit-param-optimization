import numpy as np
from .base import OptimizationAlgorithm

class GeneticAlgorithm(OptimizationAlgorithm):
    def __init__(self, evaluator, mutation_rate=0.4, elite_percentage=0.3):
        super().__init__(evaluator)
        self.param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
        self.mutation_rate = mutation_rate
        self.elite_percentage = elite_percentage
        self.bounds = np.array([
            (0.01, 50),    # alpha1
            (0.01, 50),    # alpha2
            (0.01, 50),    # alpha3
            (0.01, 50),    # alpha4
            (0.001, 100),  # delta1
            (0.001, 100),  # delta2
            (0.01, 250),   # Kd
            (1, 5)         # n
        ])

    def initialize_population(self):
        population = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1],
                                       size=(self.population_size - 1, len(self.param_names)))
        population = np.vstack((population, self.starting_point))
        return population

    def _mutate(self, individual):
        mask = np.random.rand(*individual.shape) < self.mutation_rate
        mutation_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        mutation = np.random.uniform(low=-mutation_range, high=mutation_range, size=len(self.param_names))
        individual += mask * mutation
        individual = np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])
        return individual

    def _evaluate_function(self, params):
        params_dict = dict(zip(self.param_names, params))
        return self.evaluator(**params_dict).evaluate()

    def evolve(self):
        population = self.initialize_population()
        elite_count = int(self.elite_percentage * len(population))

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

            best_individual = max(population, key=lambda x: self._evaluate_function(x))
            print(f'[{itt}] Update best with {self._evaluate_function(best_individual):.2f}')
            self.best_solution = best_individual

        return self.best_solution

    def optimize_parameters(self, generations, population_size):
        self.population_size = population_size
        self.generations = generations
        best_solution = self.evolve()
        return dict(zip(self.param_names, best_solution))
