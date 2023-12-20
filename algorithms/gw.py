import numpy as np
from .base import OptimizationAlgorithm


class GreyWolfOptimizer(OptimizationAlgorithm):
    """
    The Grey Wolf optimization algorithm implementation.
    """

    def __init__(self, evaluator):
        """
        Constructs a new GreyWolfOptimizer instance.

        Args:
            evaluator: The fitness function provider.
        """
        super().__init__(evaluator)
        # np.random.seed(42)

    def _initialize_population(self, population_size):
        """
        Initializes the whale population.

        Args:
            population_size: The population size.

        Returns:
            The randomly distributed population.
        """
        return np.random.uniform(low=np.array([bound[0] for bound in self.bounds]),
                                 high=np.array([bound[1] for bound in self.bounds]),
                                 size=(population_size, len(self.bounds)))
    def _update_position(self, current_position, alpha, beta, delta):
        """
        Updates the position based on three influential vectors (alpha, beta, delta).

        Args:
            current_position: The current position of the agent.
            alpha: Position influenced by the first vector.
            beta: Position influenced by the second vector.
            delta: Position influenced by the third vector.

        Returns:
            The updated position of the agent, constrained within the specified bounds.
        """
        a1, a2, a3 = 2 * np.random.rand(3) - 1

        c1, c2, c3 = 2 * np.random.rand(3)

        d_alpha = np.abs(c1 * alpha - current_position)
        d_beta = np.abs(c2 * beta - current_position)
        d_delta = np.abs(c3 * delta - current_position)

        new_position = alpha - a1 * d_alpha - a2 * d_beta - a3 * d_delta

        return np.clip(new_position, self.bounds[0][0], self.bounds[0][1])

    def _evaluate_function(self, params):
        """
        Evaluates the current parameters.

        Args:
            params: The parameters to evaluate.

        Returns:
            The fitness of parameters.
        """
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
        """
        Optimizes the parameters.

        Args:
            population_size: The population size.
            generations: The number of generations.

        Returns:
            The optimized parameters packed in a dictionary.
        """
        population = self._initialize_population(population_size)
        convergence_curve = []

        for generation in range(1, generations + 1):
            print(f"Generation {generation}/{generations}")
            sorted_indices = np.argsort([self._evaluate_function(w) for w in population])
            alpha, beta, delta = population[sorted_indices[:3]]

            for i in range(population_size):
                population[i] = self._update_position(population[i], alpha, beta, delta)

            convergence_curve.append(self._evaluate_function(alpha))
            print(f"Best Fitness: {convergence_curve[-1]}")

        best_params = population[np.argmin([self._evaluate_function(w) for w in population])]
        best_fitness = convergence_curve[-1] if len(convergence_curve) > 0 else np.inf
        print(f"Best Fitness: {best_fitness}")
        best_params_dict = {
            'alpha1': best_params[0],
            'alpha2': best_params[1],
            'alpha3': best_params[2],
            'alpha4': best_params[3],
            'delta1': best_params[4],
            'delta2': best_params[5],
            'Kd': best_params[6],
            'n': best_params[7]
        }
        return best_params_dict