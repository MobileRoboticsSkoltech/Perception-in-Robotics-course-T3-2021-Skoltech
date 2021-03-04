"""
Sudhanva Sreesha
ssreesha@umich.edu
24-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import numpy as np

from abc import ABC, abstractmethod


class SlamBase(ABC):
    def __init__(self, slam_type, data_association, update_type, Q):
        """
        :param slam_type: Which SLAM algorithm to run: ONLINE SLAM (ekf) or smoothing the full trajcetoryby using Factor graphs (sam).
        :param data_association: The type of data association to perform during the update step.
                                 Valid string choices include: {'known', 'nn', 'nndg', 'jcbb'}.
        :param update_type: The type of update to perform in the SLAM algorithm.
                            Valid string choices include: {'batch', 'sequential'}.
        :param Q: The observation noise covariance matrix: numpy.ndarray of size 2x2 for range and bearing measurements.
        """

        assert isinstance(slam_type, str)
        assert isinstance(data_association, str)
        assert isinstance(update_type, str)
        assert isinstance(Q, np.ndarray)

        assert slam_type in {'ekf', 'sam'}
        assert update_type in {'batch', 'sequential'}
        assert data_association in {'known', 'ml', 'jcbb'}

        self.slam_type = slam_type
        self.da_type = data_association
        self.update_type = update_type

        self.t = 0

        self.state_dim = 3  # The number of state variables: x, y, theta (initially).
        self.obs_dim = 2  # The number of variables per observation: range, bearing.
        self.lm_dim = 2  # The number of variables per landmark: x, y.

    @abstractmethod
    def predict(self, u, dt=None):
        """
        Updates mu_bar and Sigma_bar after taking a single prediction step after incorporating the control.

        :param u: The control for prediction (format: np.ndarray([drot1, dtran, drot2])).
        :param dt: The time difference between the previous state and the current state being predicted.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, z):
        """
        Performs data association to figure out previously seen landmarks vs. new landmarks
        in the observations list and updates mu and Sigma after incorporating them.

        :param z: Observation measurements (format: numpy.ndarray of size Kx3
                  observations where each row is [range, bearing, landmark_id]).
        """

        raise NotImplementedError()
