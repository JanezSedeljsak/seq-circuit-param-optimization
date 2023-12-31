from .base import EvaluationBase
from scipy.integrate import odeint
import numpy as np
from .models import three_bit_model, get_clock

class GoalEvaluator(EvaluationBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ixs_mask = np.arange(65, 1000, 120)
        matrix = np.array([
            [0, 1, 1, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0]
        ])
        self.align_matrix = matrix.reshape(matrix.shape[0], matrix.shape[1], 1)

    def single_eval(self, Q1, Q2, Q3):
        # std for preventing horizontal graphs
        q1 = Q1[self.ixs_mask, :]
        if np.std(q1) < .2 or np.std(q1[300:]) < .1:
            return -1_500_000

        # max should be bigger than 10
        q1_max = q1.max()
        if q1_max < 10:
            return -1_400_000

        # it should have atleast one wave
        if np.all(q1[:-1] < q1[1:]):
            return -1_300_000

        current = np.array([q1, Q2[self.ixs_mask, :], Q3[self.ixs_mask, :]])
        diff_scale = np.abs(q1_max - 100) ** 2
        result = -np.sum(np.abs(self.align_matrix * q1_max - current)) * max(diff_scale, 1)
        return max(result, -1_000_000)

    def evaluate(self, T=np.linspace(0, 200, 1000), **kwargs):
        """
        Search for the biggest amount of waves
        """
        params_ff = self.params
        Y0 = np.zeros(12)
        Y = odeint(three_bit_model, Y0, T, args=(params_ff,))
        Y_reshaped = np.split(Y, Y.shape[1], 1)

        Q1 = Y_reshaped[2]
        Q2 = Y_reshaped[6]
        Q3 = Y_reshaped[10]

        self.evaluator_export(Q1, Q2, Q3, **kwargs)
        return self.single_eval(Q1, Q2, Q3)
