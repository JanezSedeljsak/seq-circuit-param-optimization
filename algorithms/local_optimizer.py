import numpy as np

class LocalOptimizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator()
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

    def evaluate_params(self, params):
        self.evaluator.set_params_list(params)
        evaluation = self.evaluator.evaluate()
        return evaluation

    def search(self, params, initial_temperature=0.4, cooling_rate=0.003, num_iterations=1000):
        current_params = np.array(list(params.values()))
        best_params, best_evaluation = None, -100_000_000
        ten_percent = num_iterations // 10
        np.random.seed(42)

        for iteration in range(num_iterations):
            temperature = initial_temperature * np.exp(-cooling_rate * iteration)
            random_params = np.random.uniform(-0.5, 0.5, len(current_params)) * temperature + current_params
            random_params = np.clip(random_params, self.bounds[:, 0], self.bounds[:, 1])
            if iteration % ten_percent == 0 and iteration != 0:
                percentage = (iteration / num_iterations) * 100
                print(f"{int(percentage)}% - Best: {best_evaluation}")

            current_evaluation = self.evaluate_params(random_params)
            if current_evaluation > best_evaluation or (np.random.rand() * num_iterations * .1) < np.exp((current_evaluation - best_evaluation) / temperature):
                best_evaluation = current_evaluation
                best_params = random_params
                current_params = random_params

        param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
        best_params_dict = dict(zip(param_names, best_params))
        return best_evaluation, best_params_dict