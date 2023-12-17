from evaluators.frequency_evaluator import FreqEvaluator
from evaluators.example_evaluator import ExampleEvaluator
from algorithms.ga import GeneticAlgorithm
from algorithms.sa import SimulatedAnnealing
import matplotlib.pyplot as plt

# print(FreqEvaluator(**ExampleEvaluator().get_params()).evaluate())
# print('-'*100)

best_params = SimulatedAnnealing(FreqEvaluator).optimize_parameters()
evaluation = FreqEvaluator(**best_params)

print(evaluation.evaluate())
evaluation.simulate()