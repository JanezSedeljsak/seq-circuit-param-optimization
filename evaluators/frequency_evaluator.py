from .base import EvaluationBase
from scipy.integrate import odeint
from scipy.signal import find_peaks
import numpy as np
from .models import *

class FreqEvaluator(EvaluationBase):

    def evaluate(self, T=np.linspace(0, 200, 1000)):
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
        return np.sum([self.count_waves(Q1), self.count_waves(Q2), self.count_waves(Q3)])

    def count_waves(self, arr):
        flattened_arr = arr.flatten()
        if flattened_arr.max() - flattened_arr.min() < 20.0:
            return 0

        max_val = flattened_arr.max()
        mask_high = (flattened_arr > (max_val * 0.9))
        mask_low = (flattened_arr < (max_val * 0.1))
        potential_wave = mask_high | mask_low
        transitions = np.sum(potential_wave[:-1] != potential_wave[1:])
        wave_count = transitions // 2
        return wave_count * np.log(flattened_arr.max() - flattened_arr.min())