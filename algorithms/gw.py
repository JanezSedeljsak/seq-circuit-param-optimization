import numpy as np
from .base import OptimizationAlgorithm


class GreyWolfOptimizer(OptimizationAlgorithm):
    """
    The Grey Wolf optimization algorithm implementation.
    """

    def __init__(self, evaluator, **kwargs):
        """
        Constructs a new GreyWolfOptimizer instance.

        Args:
            evaluator: The fitness function provider.
        """
        super().__init__(evaluator, **kwargs)
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
        np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1], out=new_position)
        return new_position

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

    def optimize_parameters(self, population_size: int, generations: int, top: int = None):
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
            sorted_indices = np.argsort([-self._evaluate_function(w, export_index=generation - 1, export=self.export_data) for w in population])
            alpha, beta, delta = population[sorted_indices[:3]]

            current_eval = -self._evaluate_function(alpha)
            convergence_curve.append(current_eval)
            if best_params is None or current_eval < -self._evaluate_function(best_params):
                self.do_print(f"[{generation}] Best Fitness: {current_eval}")
                best_params = alpha.copy()

            self.generations_evaluations[generation] = current_eval

            for i in range(population_size):
                population[i] = self._update_position(population[i], alpha, beta, delta)



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
        self.do_export(top)
        return best_params_dict