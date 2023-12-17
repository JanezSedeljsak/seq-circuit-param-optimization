class OptimizationAlgorithm:
    def __init__(self, evaluator):
        self.starting_point = [34.73, 49.36, 32.73, 49.54, 1.93, 0.69, 10.44, 4.35]
        self.current_solution = self.starting_point
        self.best_solution = self.starting_point
        self.evaluator = evaluator

    def _evaluate_function(self, params):
        return self.evaluator.evaluate(params)

    def optimize_parameters(self):
        raise NotImplementedError("Subclasses must implement optimize_parameters method.")
