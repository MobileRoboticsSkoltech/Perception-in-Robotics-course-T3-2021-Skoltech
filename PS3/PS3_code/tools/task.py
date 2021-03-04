"""
Sudhanva Sreesha
ssreesha@umich.edu
21-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018

General utilities available to the filter and internal functions.
"""

import contextlib
import numpy as np
from numpy.random import normal as sample1d

from field_map import FieldMap
from matplotlib import animation as anim


def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi

    while angle < -np.pi:
        angle += pi2

    while angle >= np.pi:
        angle -= pi2

    return angle


def get_movie_writer(should_write_movie, title, movie_fps, plot_pause_len):
    """
    :param should_write_movie: Indicates whether the animation of SLAM should be written to a movie file.
    :param title: The title of the movie with which the movie writer will be initialized.
    :param movie_fps: The frame rate of the movie to write.
    :param plot_pause_len: The pause durations between the frames when showing the plots.
    :return: A movie writer that enables writing MP4 movie with the animation from SLAM.
    """

    get_ff_mpeg_writer = anim.writers['ffmpeg']
    metadata = dict(title=title, artist='matplotlib', comment='PS3: EKF SLAM')
    movie_fps = min(movie_fps, float(1. / plot_pause_len))

    return get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)


@contextlib.contextmanager
def get_dummy_context_mgr():
    """
    :return: A dummy context manager for conditionally writing to a movie file.
    """
    yield None


def get_prediction(state, motion):
    """
    Predicts the next state given state and the motion command.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param motion: The motion command to execute (format: [drot1, dtran, drot2]).
    :return: The next state of the robot after executing the motion command
             (format: np.array([x, y, theta])). The angle will be in range
             [-pi, +pi].
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(motion, np.ndarray)

    assert state.shape == (3,)
    assert motion.shape == (3,)

    x, y, theta = state
    drot1, dtran, drot2 = motion

    theta += drot1
    x += dtran * np.cos(theta)
    y += dtran * np.sin(theta)
    theta += drot2

    # Wrap the angle between [-pi, +pi].
    theta = wrap_angle(theta)

    return np.array([x, y, theta])


def sample_from_odometry(state, motion, alphas):
    """
    Predicts the next state (a noisy version) given the current state, and the motion command.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param motion: The motion command (format: [drot1, dtran, drot2]) to execute.
    :param alphas: The motion noise parameters (format: [a1, a2, a3, a4]).
    :return: A noisy version of the state prediction (format: [x, y, theta]).
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(motion, np.ndarray)

    assert state.shape == (3,)
    assert motion.shape == (3,)
    assert len(alphas) == 4

    a1, a2, a3, a4 = alphas
    drot1, dtran, drot2 = motion
    noisy_motion = np.zeros(motion.size)

    noisy_motion[0] = sample1d(drot1, np.sqrt(a1 * (drot1 ** 2) + a2 * (dtran ** 2)))
    noisy_motion[1] = sample1d(dtran, np.sqrt(a3 * (dtran ** 2) + a4 * ((drot1 ** 2) + (drot2 ** 2))))
    noisy_motion[2] = sample1d(drot2, np.sqrt(a1 * (drot2 ** 2) + a2 * (dtran ** 2)))

    return get_prediction(state, noisy_motion)


def get_observation(state, field_map, lm_id):
    """
    Generates a sample observation given the current state of the robot and the marker id of which to observe.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param field_map: A map of the field.
    :param lm_id: The landmark id indexing into the landmarks list in the field map.
    :return: The observation to the landmark (format: np.array([range, bearing, landmark_id])).
             The bearing (in rad) will be in [-pi, +pi].
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, FieldMap)

    assert state.shape == (3,)

    lm_id = int(lm_id)

    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    return np.array([distance, wrap_angle(bearing), lm_id])
    
def get_motion_noise_covariance(motion, alphas):
    """
    :param motion: The action command at the current time step (format: [drot1, dtran, drot2]).
    :param alphas: The action noise parameters (format [a1, a2, a3, a4]).
    :return: The covariance of the transition function noise (in action space).
    """

    assert isinstance(motion, np.ndarray)
    assert isinstance(alphas, np.ndarray)

    assert motion.shape == (3,)
    assert alphas.shape == (4,)

    drot1, dtran, drot2 = motion
    a1, a2, a3, a4 = alphas

    return np.diag([a1 * drot1 ** 2 + a2 * dtran ** 2,
                    a3 * dtran ** 2 + a4 * (drot1 ** 2 + drot2 ** 2),
                    a1 * drot2 ** 2 + a2 * dtran ** 2])


def get_gaussian_statistics(samples):
    """
    Computes the parameters of the samples assuming the samples are part of a Gaussian distribution.

    :param samples: The samples of which the Gaussian statistics will be computed (shape: N x 3).
    :return: Gaussian object from utils.objects with the mean and covariance initialized.
    """

    assert isinstance(samples, np.ndarray)
    assert samples.shape[1] == 3

    # Compute the mean along the axis of the samples.
    mu = np.mean(samples, axis=0)

    # Compute mean of angles.
    angles = samples[:, 2]
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mu[2] = np.arctan2(sin_sum, cos_sum)

    # Compute the samples covariance.
    mu_0 = samples - np.tile(mu, (samples.shape[0], 1))
    mu_0[:, 2] = np.array([wrap_angle(angle) for angle in mu_0[:, 2]])
    Sigma = mu_0.T @ mu_0 / samples.shape[0]

    return Gaussian(mu, Sigma)
