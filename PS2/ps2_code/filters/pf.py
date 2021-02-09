"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)
        
        # TODO add here specific class variables for the PF

    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        self._state_bar.mu = self.mu
        self._state_bar.Sigma = self.Sigma

    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
