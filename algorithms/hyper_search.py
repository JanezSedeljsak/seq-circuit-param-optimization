from collections import defaultdict
from tqdm import tqdm_notebook
import itertools
import numpy as np

class HyperSearch:
    def __init__(self, algorithm, evaluator, is_export=False):
        self.algorithm = algorithm
        self.evaluator = evaluator()
        self._evaluator = evaluator
        self.is_export = is_export
        self.top5_params = []

    def execute(self, generations=10, **params):
        scores = defaultdict(list)
        seeds = [1, 42, 421, 555]

        best_seed = None
        best_max_score = float('-inf')
        best_param_set = None

        keys = list(params.keys())
        param_combinations = [
            dict(zip(keys, values)) for values in itertools.product(*params.values())
        ]

        total_steps = len(param_combinations) * len(seeds)
        print(f'Hyper search start [steps={total_steps}]')
        has_converged = False

        for i, seed in tqdm_notebook(enumerate(seeds), total=len(seeds)):
            if has_converged:
                break

            for j, param_set in tqdm_notebook(enumerate(param_combinations), total=len(param_combinations)):
                np.random.seed(seed)
                population = param_set.get("population", 10)
                evaluator_params = self.algorithm(self._evaluator, is_print=False, **param_set).optimize_parameters(population, generations)
                self.evaluator.set_params(**evaluator_params)
                score = self.evaluator.evaluate()
                param_key = tuple(sorted(param_set.items()))
                scores[param_key].append(score)

                max_scores = {param_key: np.max(score_list) for param_key, score_list in scores.items()}
                current_best_max_score = max(max_scores.values())
                # print(f'Done with {i * len(param_combinations) + j + 1}/{total_steps}')

                if current_best_max_score > best_max_score:
                    best_max_score = current_best_max_score
                    best_param_set = max(max_scores, key=max_scores.get)
                    best_seed = seed

                # Update top 5 params
                self.top5_params.append((best_max_score, best_param_set))

                # starting to converge
                if best_max_score > -200:
                    print(f'Converged after {i * len(param_combinations) + j + 1}')
                    has_converged = True
                    break

        self.best_params = dict(best_param_set)
        self.best_seed = best_seed
        self.best_evluation = best_max_score
        self.top5_params = sorted(self.top5_params, key=lambda x: x[0], reverse=True)[:5]
        return self

    def optimize_parameters(self, population_size: int, generations: int, seed: int = None, top: int = None):
        np.random.seed(self.best_seed if seed is None else seed)
        best_params = self.algorithm(self._evaluator, is_export=self.is_export, **self.best_params).optimize_parameters(population_size, generations, top)
        self.evaluator.set_params(**best_params)
        return self.evaluator
    
    def optimize_top5(self, population_size: int, generations: int, seed: int = None) -> list:
        """
        Returns a list of evaluators which contain the top 5 best parameter combinations
        """
        optimized_top5 = []
        count = 0
        for _, params in self.top5_params:
            count += 1
            print("This algorithm is: ", self.algorithm.__class__.__name__)
            if self.algorithm.__class__.__name__ == 'GeneticAlgorithm':
                best_params = self.algorithm(self._evaluator, is_export=self.is_export, **dict(params)).optimize_parameters(population_size, generations, seed=seed, top=count)
            else:
                best_params = self.algorithm(self._evaluator, is_export=self.is_export, **dict(params)).optimize_parameters(population_size, generations, top=count)
            # Duplicate the evaluator, set the params and store it in the list
            evaluator = self._evaluator()
            evaluator.set_params(**best_params)
            optimized_top5.append(evaluator)
        return optimized_top5 
