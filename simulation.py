from evaluators.frequency_evaluator import FreqEvaluator
from evaluators.example_evaluator import ExampleEvaluator
from algorithms.ga import GeneticAlgorithm
from algorithms.sa import SimulatedAnnealing
import matplotlib.pyplot as plt

# print(FreqEvaluator(**ExampleEvaluator().get_params()).evaluate())
# print('-'*100)
# evaluation = FreqEvaluator(**ExampleEvaluator().get_params())

best_params = GeneticAlgorithm(FreqEvaluator).optimize_parameters(50, 50)
evaluation = FreqEvaluator(**best_params)

print(evaluation.evaluate())
evaluation.simulate()