from algorithms import *
from evaluators import *
import numpy as np
import sys

# print(FreqEvaluator(**ExampleEvaluator().get_params()).evaluate())
# print('-'*100)
# evaluation = FreqEvaluator(**ExampleEvaluator().get_params())

# print(ClockEvaluator(**ExampleEvaluator().get_params()).evaluate())
# print('-'*100)
# evaluation = ClockEvaluator(**ExampleEvaluator().get_params())

POPULATION_SIZE = 10
GENERATIONS = 20
IS_EXPORT = True
EVALUATOR = GoalEvaluator

np.random.seed(42)

if sys.argv and len(sys.argv) > 1:
    method = sys.argv[1]

    if method == 'ga':
        best_params = GeneticAlgorithm(EVALUATOR, is_export=IS_EXPORT).optimize_parameters(POPULATION_SIZE, GENERATIONS)
        evaluation = EVALUATOR(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    elif method == 'woa':
        best_params = WhaleOptimizationAlgorithm(EVALUATOR, is_export=IS_EXPORT).optimize_parameters(POPULATION_SIZE, GENERATIONS)
        evaluation = EVALUATOR(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    elif method == 'gw':
        best_params = GreyWolfOptimizer(EVALUATOR, is_export=IS_EXPORT).optimize_parameters(POPULATION_SIZE, GENERATIONS)
        evaluation = EVALUATOR(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    elif method == 'ant':
        best_params = AntColonyOptimization(EVALUATOR, is_export=IS_EXPORT).optimize_parameters(POPULATION_SIZE, GENERATIONS)
        evaluation = EVALUATOR(**best_params)
        print(evaluation.evaluate())
        evaluation.simulate()
    else:
        print('Invalid method')
else:
    best_params = WhaleOptimizationAlgorithm(EVALUATOR, is_export=IS_EXPORT).optimize_parameters(POPULATION_SIZE, GENERATIONS)
    evaluation = EVALUATOR(**best_params)
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


