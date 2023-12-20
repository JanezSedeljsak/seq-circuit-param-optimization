from .base import OptimizationAlgorithm
import numpy as np


class WhaleOptimizationAlgorithm(OptimizationAlgorithm):
    """
    The Whale optimization algorithm implementation.
    """

    def __init__(self, evaluator):
        """
        Constructs a new WhaleOptimizationAlgorithm instance.

        Args:
            evaluator: The fitness function provider.
        """
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
        # Set seed for reproducibility
        # np.random.seed(42)
        # np.random.seed(444)
        # np.random.seed(666)
        # np.random.seed(5)
        # np.random.seed(23)
        # np.random.seed(52)

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

    def _update_position(self, current_position, leader_position, A):
        """
        Updates the position based on the leader whale.

        Args:
            current_position: The current position of the whale.
            leader_position: The position of the leader whale.
            A: A parameter affecting the update calculation.

        Returns:
            The updated position of the whale, constrained within the specified bounds.
        """
        r1 = np.random.rand()
        r2 = np.random.rand()

        A1 = 2 * A * r1 - A
        C1 = 2 * r2 - 1

        distance_to_leader = np.abs(C1 * leader_position - current_position)
        new_position = leader_position - A1 * distance_to_leader

        return np.clip(new_position, self.bounds[0][0], self.bounds[0][1])

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
            leader_position = population[np.argmin([self._evaluate_function(w) for w in population])]

            for i in range(population_size):
                a = 2 - 2 * generation / generations  # linearly decreases from 2 to 0
                A = 2 * a * np.random.rand() - a

                population[i] = self._update_position(population[i], leader_position, A)

            convergence_curve.append(self._evaluate_function(leader_position))
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