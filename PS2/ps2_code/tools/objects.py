"""
Sudhanva Sreesha
ssreesha@umich.edu
10-Apr-2018

Gonzalo Ferrer,
g.ferrer@skoltech.ru
"""

import numpy as np


class Gaussian(object):
    """
    Represents a multi-variate Gaussian distribution representing the state of the robot.
    """

    def __init__(self, mu, Sigma):
        """
        Sets the internal mean and covariance of the Gaussian distribution.

        :param mu: A 1-D numpy array (size 3x1) of the mean (format: [x, y, theta]).
        :param Sigma: A 2-D numpy ndarray (size 3x3) of the covariance matrix.
        """

        assert isinstance(mu, np.ndarray)
        assert isinstance(Sigma, np.ndarray)
        assert Sigma.shape == (3, 3)

        if mu.ndim < 1:
            raise ValueError('The mean must be a 1D numpy ndarray of size 3.')
        elif mu.shape == (3,):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            mu = mu[np.newaxis].T
        elif mu.shape != (3, 1):
            raise ValueError('The mean must be a vector of size 3x1.')

        self.mu = mu
        self.Sigma = Sigma


class FilterInputData(object):
    """
    Represents the data that is available to the filter while estimating the robot state.
    """

    def __init__(self, motion_commands, observations):
        """
        Sets the internal data available to robot state estimation filter.
        Let N be the number of steps in the robot state estimation simulation.

        :param motion_commands: A 2-D numpy ndarray of size Nx3 where each row is [drot1, dtran, drot2].
        :param observations: A 2-D numpy ndarray of size Nx3 where each row is [bearing, landmark_id].
        """

        assert isinstance(motion_commands, np.ndarray)

        assert motion_commands.ndim >= 2 and motion_commands.shape[1] == 3
        assert observations.ndim >= 2 and observations.shape[1] == 2

        self.motion_commands = motion_commands
        self.observations = observations


class FilterDebugData(object):
    """
    Contains data only available for debugging/displaying purposes during robot state estimation.
    """

    def __init__(self, real_robot_path, noise_free_robot_path, noise_free_observations):
        """
        Sets the internal data only available for debugging purposes to the state estimation filter.
        Let N be the number of steps in the robot state estimation filter.

        :param real_robot_path: A 2-D numpy ndarray of size Nx3 where each row is [x, y, theta].
        :param noise_free_robot_path: A 2-D numpy ndarray of size Nx3 where each row is [x, y, theta].
        :param noise_free_observations: A 2-D numpy ndarray of size Nx2 where each row is [bearing, landmark_id].
        """

        assert isinstance(real_robot_path, np.ndarray)
        assert isinstance(noise_free_robot_path, np.ndarray)
        assert isinstance(noise_free_observations, np.ndarray)

        assert real_robot_path.ndim >= 2 and real_robot_path.shape[1] == 3
        assert noise_free_robot_path.ndim >= 2 and noise_free_robot_path.shape[1] == 3
        assert noise_free_observations.ndim >= 2 and noise_free_observations.shape[1] == 2

        self.real_robot_path = real_robot_path
        self.noise_free_robot_path = noise_free_robot_path
        self.noise_free_observations = noise_free_observations


class SimulationData(object):
    """
    Contains all data necessary to run the robot state estimation simulation.
    """

    def __init__(self, num_steps, filter_data, debug_data):
        """
        Initializes the internal variables.

        :param num_steps: A scalar value representing the number of steps to run the estimation filter for.
        :param filter_data: A FilterInputData object.
        :param debug_data: A FilterDebugData object.
        """

        assert isinstance(num_steps, int)
        assert isinstance(filter_data, FilterInputData)
        assert isinstance(debug_data, FilterDebugData)

        self.num_steps = num_steps
        self.filter = filter_data
        self.debug = debug_data


class FilterTrajectory(object):
    def __init__(self, mean_trajectory, covariance_trajectory=None):
        """
        Contains trajectories estimated by the localization filter for the input data.
        Let N be the number of time steps and M be the number of particles (only for the PF).

        :param mean_trajectory: The mean path estimated by the localization filter (per time-step).
                                For the EKF, this is a numpy.ndarray of shape Nx3. Whereas,
                                for the PF, this is of shape Nx3xM.
        :param covariance_trajectory: The covariance of the estimate at each time-step.
                                      A numpy.ndarray of shape 3x3xN. Since the covariance
                                      trajectory might not always be necessary, it's optional.
        """

        assert isinstance(mean_trajectory, np.ndarray)
        assert mean_trajectory.ndim >= 2 and mean_trajectory.shape[1] == 3

        self.mean = mean_trajectory

        if covariance_trajectory is None:
            return

        assert isinstance(covariance_trajectory, np.ndarray)
        assert covariance_trajectory.ndim == 3 and covariance_trajectory.shape[:-1] == (3, 3)

        self.covariance = covariance_trajectory
