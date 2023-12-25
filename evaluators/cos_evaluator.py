from .goal_evaluator import GoalEvaluator
from scipy.integrate import odeint
import numpy as np
from .models import three_bit_model, get_clock

class CosineEvaluator(GoalEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ixs_mask = np.arange(50, 1000, 120)
        self.align_matrix = self.align_matrix * 100.0
        self.norm_align_matrix = np.linalg.norm(self.align_matrix)
        self.flat_algin_matrix = self.align_matrix.flatten()

    def single_eval(self, Q1, Q2, Q3):
        current = np.array([Q1[self.ixs_mask, :].flatten(),
                            Q2[self.ixs_mask, :].flatten(),
                            Q3[self.ixs_mask, :].flatten()])

        current_flat = current.flatten()
        similarity_scores = np.dot(current_flat, self.flat_algin_matrix)
        norms = np.linalg.norm(current_flat) * self.norm_align_matrix
        return similarity_scores / norms
