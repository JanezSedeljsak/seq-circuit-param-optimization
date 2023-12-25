import numpy as np
from joblib import Parallel, delayed

def _optimize_algorithm(algorithm, population, generations):
    return algorithm.optimize_parameters(population, generations)

def MultiOptimizer(*algorithms, population=10, generations=10):
    results = Parallel(n_jobs=-1)(
        delayed(_optimize_algorithm)(algorithm, population, generations) for algorithm in algorithms
    )

    base_algorithm, *_ = algorithms
    new_init_population = np.array([list(individual.values()) for individual in results])
    ixs = np.argsort([-base_algorithm._evaluate_function(individual) for individual in new_init_population])
    best_base_population = new_init_population[ixs[:population]]
    return base_algorithm.optimize_parameters(population, generations, base_population=best_base_population)

    

