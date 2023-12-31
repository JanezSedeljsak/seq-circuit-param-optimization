import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.integrate import odeint
from .models import *
from .params import *

Params = namedtuple('Params', ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n'])

class EvaluationBase:
    standardization = None
    standardization_base = None
    starting_point = [34.73, 49.36, 32.73, 49.54, 1.93, 0.69, 10.44, 4.35]
    """
    Base class for models.

    Attributes:
        Params (namedtuple): Structure to hold model parameters.
        params (Params): Model parameters.
    """

    def __init__(self, **kwargs):
        """
        Initializes the EvaluationBase object.

        Args:
            **kwargs: Keyword arguments to set the initial model parameters.
        """
        self.params = Params(**kwargs) if kwargs else \
            Params(alpha1=0, alpha2=0, alpha3=0, alpha4=0, delta1=0, delta2=0, Kd=0, n=0)

    def set_params(self, **kwargs):
        """
        Sets the model parameters.

        Args:
            **kwargs: Keyword arguments to update the model parameters.
        """
        params_dict = self.params._asdict()
        self.params = Params(**{**params_dict, **kwargs})

    def set_params_list(self, params):
        self.params = Params._make(params)

    def evaluate(self, weights=[], joined=[]):
        """
        Evaluates the model.
        """
        raise MethodNotImplementedError()

    def search(self):
        """
        Searches for optimal model parameters.
        Should update the model parameters within the method.
        """
        raise MethodNotImplementedError()

    def evaluator_export(self, Q1, Q2, Q3, export=None, export_index=-1, **kwargs):
        if export and export_index != -1:
            export[export_index] = np.concatenate((export[export_index], Q1.flatten(), Q2.flatten(), Q3.flatten()))

    def simulate(self, N=1000, t_end=200, out=None, algo=''):
        """
        Simulates the model.

        Args:
            N (int): Number of time steps.
            t_end (float): End time of simulation.
        """
        params_ff = self.params

        # three-bit counter with external clock
        Y0 = np.array([0] * 12)  # initial state
        T = np.linspace(0, t_end, N)  # vector of timesteps

        # numerical interation
        Y = odeint(three_bit_model, Y0, T, args=(params_ff,))

        Y_reshaped = np.split(Y, Y.shape[1], 1)

        # plotting the results
        Q1 = Y_reshaped[2]
        not_Q1 = Y_reshaped[3]
        Q2 = Y_reshaped[6]
        not_Q2 = Y_reshaped[7]
        Q3 = Y_reshaped[10]
        not_Q3 = Y_reshaped[11]

        plt.clf() # clear plot
        plt.style.use('ggplot')
        if algo: plt.title(f'{algo[0].upper() + algo[1:]} - {self.__class__.__name__}')
        plt.plot(T, Q1, label='q1', alpha=0.7)
        plt.plot(T, Q2, label='q2', alpha=0.7)
        plt.plot(T, Q3, label='q3', alpha=0.7)
        # plt.plot(T, not_Q1, label='not q1')
        # plt.plot(T, not_Q2, label='not q2')

        plt.plot(T, get_clock(T), '--', linewidth=2, label="CLK", color='black', alpha=0.25)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()
        if out is not None:
            plt.savefig(out)


    def get_params(self):
        """
        Returns the model parameters.

        Returns:
            dict: Dictionary of model parameters.
        """
        return self.params._asdict()

    @classmethod
    def get_standardization(cls):
        bounds = np.array([
            (0.01, 50),  # alpha1
            (0.01, 50),  # alpha2
            (0.01, 50),  # alpha3
            (0.01, 50),  # alpha4
            (0.001, 100),  # delta1
            (0.001, 100),  # delta2
            (0.01, 250),  # Kd
            (1, 5)  # n
        ])
        if cls.standardization is not None:
            return cls.standardization

        if cls.standardization_base is None:
            cls.standardization_base = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(100, 8))

        param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
        evaluators = [cls(**dict(zip(param_names, params))) for params in cls.standardization_base]
        results = [evaluator.evaluate() for evaluator in evaluators]
        baseline_score = cls(**dict(zip(param_names, cls.starting_point))).evaluate()
        cls.standardization = np.mean(results), np.std(results), baseline_score
        return cls.standardization

    @staticmethod
    def vector_converges(vector, threshold):
        final_value = vector[-1]
        converged_values = vector[np.abs(vector - final_value) <= threshold]
        return converged_values.shape[0] >= 0.8 * len(vector)