class OptimizationAlgorithm:
    def __init__(self, evaluator, joined=[], weights=[]):
        self.starting_point = [34.73, 49.36, 32.73, 49.54, 1.93, 0.69, 10.44, 4.35]
        self.current_solution = self.starting_point
        self.best_solution = self.starting_point
        self.evaluator = evaluator
        self.joined = joined
        self.weigths = weights
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

    def _evaluate_function(self, params):
        return self.evaluator.evaluate(params)

    def optimize_parameters(self):
        raise NotImplementedError("Subclasses must implement optimize_parameters method.")
