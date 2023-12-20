from .base import EvaluationBase
from scipy.integrate import odeint
import numpy as np
from .models import *

class JoinedEvaluator(EvaluationBase):

    def __init__(self, *evaluators, **kwargs):
        super().__init__(**kwargs)
        self.evaluators = evaluators

    def evaluate(self, T=np.linspace(0, 200, 1000)):
        """
        Caclulate the mean of the standardized evaluators
        """
        params_ff = self.params
        Y0 = np.zeros(12)
        Y = odeint(three_bit_model, Y0, T, args=(params_ff,))
        Y_reshaped = np.split(Y, Y.shape[1], 1)

        Q1 = Y_reshaped[2]
        Q2 = Y_reshaped[6]
        Q3 = Y_reshaped[10]
        C = get_clock(T)

        scores = []
        for evaluator in self.evaluators:
            current_score = evaluator.single_eval(Q1, Q2, Q3, C)
            randoms_mean, randoms_std, baseline = evaluator.get_standardization()

            scaled_score = (current_score - randoms_mean) / (baseline - randoms_mean)
            scaled_score += scaled_score * (np.log(randoms_std + 1) / 100)
            scores.append(scaled_score)

        return np.sum(scores)