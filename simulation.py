from algorithms.ant import AntColonyOptimization
from algorithms.ga import GeneticAlgorithm
from algorithms.gw import GreyWolfOptimizer
from algorithms.woa import WhaleOptimizationAlgorithm
from evaluators.clock_evaluator import ClockEvaluator
from evaluators.example_evaluator import ExampleEvaluator
from evaluators.frequency_evaluator import FreqEvaluator
from evaluators.joined_evaluator import JoinedEvaluator
import sys

print(FreqEvaluator(**ExampleEvaluator().get_params()).evaluate())
print('-'*100)
# evaluation = FreqEvaluator(**ExampleEvaluator().get_params())

print(ClockEvaluator(**ExampleEvaluator().get_params()).evaluate())
print('-'*100)
# evaluation = ClockEvaluator(**ExampleEvaluator().get_params())

POPULATION_SIZE = 10
GENERATIONS = 10

if sys.argv and len(sys.argv) > 1:
    method = sys.argv[1]

    if method == 'ga':
        best_params = GeneticAlgorithm(FreqEvaluator).optimize_parameters(POPULATION_SIZE, GENERATIONS)
        evaluation = FreqEvaluator(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    elif method == 'woa':
        best_params = WhaleOptimizationAlgorithm(FreqEvaluator).optimize_parameters(POPULATION_SIZE, GENERATIONS)
        evaluation = FreqEvaluator(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    elif method == 'gw':
        best_params = GreyWolfOptimizer(FreqEvaluator).optimize_parameters(POPULATION_SIZE, GENERATIONS)
        evaluation = FreqEvaluator(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    elif method == 'ant':
        best_params = AntColonyOptimization(FreqEvaluator).optimize_parameters()
        evaluation = FreqEvaluator(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    else:
        print('Invalid method')
else:
    best_params = WhaleOptimizationAlgorithm(FreqEvaluator).optimize_parameters(POPULATION_SIZE, GENERATIONS)
    evaluation = FreqEvaluator(**best_params)
    print(evaluation.evaluate())
    evaluation.simulate()

    #best_params = WhaleOptimizationAlgorithm(ClockEvaluator).optimize_parameters(POPULATION_SIZE, GENERATIONS)
    #evaluation = FreqEvaluator(**best_params)
    #print(evaluation.evaluate())
    #evaluation.simulate()

    # best_params = WhaleOptimizationAlgorithm(JoinedEvaluator).optimize_parameters(POPULATION_SIZE, GENERATIONS)
    # evaluation = FreqEvaluator(**best_params)
    # print(evaluation.evaluate())
    # evaluation.simulate()

    # best_params = GreyWolfOptimizer(JoinedEvaluator).optimize_parameters(POPULATION_SIZE, GENERATIONS)
    # evaluation = FreqEvaluator(**best_params)
    # print(evaluation.evaluate())
    # evaluation.simulate()

    # best_params = (AntColonyOptimization(JoinedEvaluator, num_ants=POPULATION_SIZE, max_iterations=GENERATIONS)
    #                .optimize_parameters())
    # evaluation = FreqEvaluator(**best_params)
    # print(evaluation.evaluate())
    # evaluation.simulate()


