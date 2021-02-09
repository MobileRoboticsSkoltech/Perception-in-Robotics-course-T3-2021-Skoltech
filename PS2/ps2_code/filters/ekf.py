"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
