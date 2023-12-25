from evaluators.helpers import *
import numpy as np

T = np.linspace(0, 200, 1000)
C = get_clock(T)
np.savetxt(f'drawer/data/export_CLK.csv', C, delimiter=',', fmt='%.3f')
