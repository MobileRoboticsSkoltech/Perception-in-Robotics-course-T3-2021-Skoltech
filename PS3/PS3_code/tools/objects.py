"""
Sudhanva Sreesha
ssreesha@umich.edu
10-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
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

        if not isinstance(mu, np.ndarray):
            raise TypeError('mu should be of type np.ndarray.')

        if mu.ndim < 1:
            raise ValueError('The mean must be a 1D numpy ndarray of size 3.')
        elif mu.shape == (3,):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            mu = mu[np.newaxis].T
        elif mu.shape != (3, 1):
            raise ValueError('The mean must be a vector of size 3x1.')
        if not isinstance(Sigma, np.ndarray):
            raise TypeError('Sigma should be of type np.ndarray.')

        self.mu = mu
        self.Sigma = Sigma

class SlamInputData(object):
    """
    Represents the data that is available to the SLAM algorithm while estimating the robot state.
    """

    def __init__(self, motion_commands, observations):
        """
        Sets the internal data available to SLAM algorithm while estimating the world state.

        Let M be the number of observations sensed per time step in the simulation.
        Let N be the number of steps in the robot state estimation simulation.

        :param motion_commands: A 2-D numpy ndarray of size Nx3 where each row is [drot1, dtran, drot2].
        :param observations: A 3-D numpy ndarray of size NxMx3 where observations are of format: [range (cm, float),
                                                                                                  bearing (rad, float),
                                                                                                  landmark_id (id, int)]
        """

        if not isinstance(motion_commands, np.ndarray):
            raise TypeError('motion_commands should be of type np.ndarray.')

        if motion_commands.ndim != 2 or motion_commands.shape[1] != 3:
            raise ValueError('motion_commands should be of size Nx3 where N is the number of time steps.')

        if observations.ndim != 3 or observations.shape[2] != 3:
            raise ValueError('Observations should be of size NxMx3 where M is the number '
                             'of observations per time step N is the number of time steps.')

        self.motion_commands = motion_commands
        self.observations = observations


class SlamDebugData(object):
    """
    Contains data only available for debugging/displaying purposes during robot state estimation.
    """

    def __init__(self, real_robot_path, noise_free_robot_path, noise_free_observations):
        """
        Sets the internal data only available for debugging purposes to the state estimation filter.

        Let M be the number of observations sensed per time step in the simulation.
        Let N be the number of steps in the robot state estimation simulation.

        :param real_robot_path: A 2-D numpy ndarray of size Nx3 where each row is [x, y, theta].
        :param noise_free_robot_path: A 2-D numpy ndarray of size Nx3 where each row is [x, y, theta].
        :param noise_free_observations: A 3-D numpy ndarray of size NxMx3 where observations are of format:
                                        [range (cm, float),
                                         bearing (rad, float),
                                         landmark_id (id, int)]
        """

        if real_robot_path.ndim != 2 or real_robot_path.shape[1] != 3:
            raise ValueError('real_robot_path should be of size 3xN where N is the number of time steps.')

        if noise_free_robot_path.ndim != 2 or noise_free_robot_path.shape[1] != 3:
            raise ValueError('noise_free_robot_path should be of size 3xN where N is the number of time steps.')

        if not noise_free_observations.shape or \
                noise_free_observations.ndim != 3 or \
                noise_free_observations.shape[2] != 3:
            raise ValueError('noise_free_observations should be of size NxMx3 where M is the number '
                             'of observations per time step N is the number of time steps in the sim.')

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
        :param filter_data: A SlamInputData object.
        :param debug_data: A SlamDebugData object.
        """

        assert isinstance(num_steps, int)
        assert isinstance(filter_data, SlamInputData)
        assert isinstance(debug_data, SlamDebugData)

        self.num_steps = num_steps
        self.filter = filter_data
        self.debug = debug_data
