import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.integrate import odeint
from .models import *
from .params import *

Params = namedtuple('Params', ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n'])

class EvaluationBase:
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
        self.params = Params(**kwargs)

    def set_params(self, **kwargs):
        """
        Sets the model parameters.

        Args:
            **kwargs: Keyword arguments to update the model parameters.
        """
        params_dict = self.params._asdict()
        self.params = Params(**{**params_dict, **kwargs})

    def evaluate(self):
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

    def simulate(self, N=1000, t_end=200, out=None):
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
        plt.plot(T, Q1, label='q1')
        plt.plot(T, Q2, label='q2')
        plt.plot(T, Q3, label='q3')
        # plt.plot(T, not_Q1, label='not q1')
        # plt.plot(T, not_Q2, label='not q2')

        plt.plot(T, get_clock(T), '--', linewidth=2, label="CLK", color='black', alpha=0.25)
        plt.legend()

        if out is None: plt.show()
        else: plt.savefig(out)


    def get_params(self):
        """
        Returns the model parameters.

        Returns:
            dict: Dictionary of model parameters.
        """
        return self.params._asdict()
