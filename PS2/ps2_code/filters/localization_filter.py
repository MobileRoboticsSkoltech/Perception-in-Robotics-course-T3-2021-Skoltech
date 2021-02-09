"""
An abstract base class to implement the various localization filters in the task: EKF or  PF.
"""

from abc import ABC
from abc import abstractmethod

import numpy as np

from tools.objects import Gaussian
from field_map import FieldMap


class LocalizationFilter(ABC):
    def __init__(self, initial_state, alphas, beta):
        """
        Initializes the filter parameters.

        :param initial_state: The Gaussian distribution representing the robot prior.
        :param alphas: A 1-d np-array of motion noise parameters (format: [a1, a2, a3, a4]).
        :param beta: A scalar value of the measurement noise parameter (format: rad).
        """

        assert isinstance(initial_state, Gaussian)
        assert initial_state.Sigma.shape == (3, 3)

        if not isinstance(initial_state, Gaussian):
            raise TypeError('The initial_state must be of type `Gaussian`. (see tools/objects.py)')

        if initial_state.mu.ndim < 1:
            raise ValueError('The initial mean must be a 1D numpy ndarray of size 3.')
        elif initial_state.mu.shape == (3, ):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            initial_state.mu = initial_state.mu[np.newaxis].T
        elif initial_state.mu.shape != (3, 1):
            raise ValueError('The initial state mean must be a vector of size 3x1')

        self.state_dim = 3   # [x, y, theta]
        self.motion_dim = 3  # [drot1, dtran, drot2]
        self.obs_dim = 1     # [bearing]

        self._state = initial_state
        self._state_bar = initial_state

        # Filter noise parameters.
        self._alphas = alphas
        # Measurement variance.
        self._Q = beta ** 2

        # Setup the field map.
        self._field_map = FieldMap()

    @abstractmethod
    def predict(self, u):
        """
        Updates mu_bar and Sigma_bar after taking a single prediction step after incorporating the control.

        :param u: The control for prediction (format: [drot1, dtran, drot2]).
        """
        raise NotImplementedError('Must implement a prediction step for the filter.')

    @abstractmethod
    def update(self, z):
        """
        Updates mu and Sigma after incorporating the observation z.

        :param z: Observation measurement (format: [bearing, marker_id]).
        """
        raise NotImplementedError('Must implement an update step for the filter.')

    @property
    def mu_bar(self):
        """
        :return: The state mean after the prediction step (format: 1D array for easy indexing).
        """
        return self._state_bar.mu.T[0]

    @property
    def Sigma_bar(self):
        """
        :return: The state covariance after the prediction step (shape: 3x3).
        """
        return self._state_bar.Sigma

    @property
    def mu(self):
        """
        :return: The state mean after the update step (format: 1D array for easy indexing).
        """
        return self._state.mu.T[0]

    @property
    def Sigma(self):
        """
        :return: The state covariance after the update step (shape: 3x3).
        """
        return self._state.Sigma
