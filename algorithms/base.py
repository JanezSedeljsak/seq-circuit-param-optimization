import numpy as np

class OptimizationAlgorithm:

    def __init__(self, evaluator, joined=[], weights=[], is_export=False, is_print=True, **kwargs):
        self.export_data = None
        self.generations_evaluations = dict()
        self.is_export = is_export
        self.evaluator = evaluator()
        self.joined = joined
        self.weigths = weights
        self.do_print = print if is_print else lambda x: None
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

    def _evaluate_function(self, params, **kwargs):
        return self.evaluator.evaluate(params)

    def optimize_parameters(self):
        raise NotImplementedError("Subclasses must implement optimize_parameters method.")
    
    def create_export_matrix(self, n):
        if self.is_export:
            self.export_data = [np.ones((0))] * n

    def do_export(self, top=None):
        if self.is_export:
            np.savetxt(f'drawer/data/export_{self.__class__.__name__}.csv', self.export_data, 
                       delimiter=',', fmt='%.3f')
            if top is not None:
                print("Exporting to file: ", f'drawer/data/export_{self.__class__.__name__}_generations_{top}.csv')
                np.savetxt(f'drawer/data/export_{self.__class__.__name__}_generations_{top}.csv', 
                        np.array(list(self.generations_evaluations.items())), delimiter=',', fmt='%.3f')
