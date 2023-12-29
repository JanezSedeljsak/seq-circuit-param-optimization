from .base import OptimizationAlgorithm
import numpy as np


class WhaleOptimizationAlgorithm(OptimizationAlgorithm):
    """
    The Whale optimization algorithm implementation.
    """

    def __init__(self, evaluator, mutation_strength=0.1, **kwargs):
        """
        Constructs a new WhaleOptimizationAlgorithm instance.

        Args:
            evaluator: The fitness function provider.
        """
        super().__init__(evaluator, **kwargs)
        self.mutation_strength = mutation_strength
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

    def _evaluate_function(self, params, **kwargs):
        """
        Evaluates the current parameters.

        Args:
            params: The parameters to evaluate.

        Returns:
            The fitness of parameters.
        """
        self.evaluator.set_params_list(params)
        evaluation = self.evaluator.evaluate(**kwargs)
        return evaluation

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
        perturbation = self.mutation_strength * np.random.randn(*new_position.shape)
        new_position += perturbation

        np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1], out=new_position)
        return new_position

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
        self.create_export_matrix(generations)
        convergence_curve = []
        best_params = None

        for generation in range(1, generations + 1):
            self.do_print(f"Generation {generation}/{generations}")
            leader_position = population[np.argmin([-self._evaluate_function(w, export_index=generation - 1, export=self.export_data) for w in population])]

            current_eval = -self._evaluate_function(leader_position)
            convergence_curve.append(current_eval)
            self.do_print(f"Best Fitness: {current_eval}")
            if best_params is None or current_eval < -self._evaluate_function(best_params):
                best_params = leader_position

            for i in range(population_size):
                a = 2 - 2 * generation / generations  # linearly decreases from 2 to 0
                A = 2 * a * np.random.rand() - a

                population[i] = self._update_position(population[i], leader_position, A)

        best_fitness = -self._evaluate_function(best_params)
        self.do_print(f"Best Fitness: {best_fitness}")
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
        self.do_export()
        return best_params_dict