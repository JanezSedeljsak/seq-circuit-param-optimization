from .base import EvaluationBase
from scipy.integrate import odeint
from scipy.signal import find_peaks
import numpy as np
from .models import *

class ClockEvaluator(EvaluationBase):

    @classmethod
    def single_eval(cls, Q1, Q2, Q3, C):
        # prevent horizontal convergence
        if all(EvaluationBase.vector_converges(q, 0.1) for q in (Q1, Q2, Q3)):
            return -10e9

        return -np.sum([np.abs(Q1 - C), np.abs(Q2 - C), np.abs(Q3 - C)])

    def evaluate(self, T=np.linspace(0, 200, 1000), **kwargs):
        """
        Search for smallest diff between the clock and the exit signals
        """
        params_ff = self.params
        Y0 = np.zeros(12)
        Y = odeint(three_bit_model, Y0, T, args=(params_ff,))
        Y_reshaped = np.split(Y, Y.shape[1], 1)

        Q1 = Y_reshaped[2]
        Q2 = Y_reshaped[6]
        Q3 = Y_reshaped[10]
        C = get_clock(T)
        self.evaluator_export(Q1, Q2, Q3, **kwargs)
        return self.single_eval(Q1, Q2, Q3, C)
