#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
21-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import os

import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal as sample2d
from tools.plot import plot_robot
from field_map import FieldMap
from tools.objects import SimulationData
from tools.objects import SlamDebugData
from tools.objects import SlamInputData
from tools.plot import plot_field
from tools.plot import plot_observations
from tools.task import get_observation
from tools.task import sample_from_odometry


def generate_motion(t, dt):
    """
    Generates a square motion.

    :param t: Time (in seconds) for the current time step.
    :param dt: Time increment (in seconds) between consecutive steps.

    :raises ValueError if dt > 1.0
    :return: [first rotation (rad), forward distance, second rotation (rad)]
    """

    assert dt <= 1.0

    n = t / dt
    hz = 1 / dt
    i = np.mod(n, np.floor(hz) * 5)

    if i == 0:
        u = np.array([0, dt * 100, 0])
    elif i == 1 * hz:
        u = np.array([0, dt * 100, 0])
    elif i == 2 * hz:
        u = [np.deg2rad(45), dt * 100, np.deg2rad(45)]
    elif i == 3 * hz:
        u = [0, dt * 100, 0]
    elif i == 4 * hz:
        u = [np.deg2rad(45), 0.1, np.deg2rad(45)]
    else:
        u = np.array([0, dt * 100, 0])

    return u


def sense_landmarks(state, field_map, max_observations):
    """
    Observes num_observations of landmarks for the current time step.
    The observations will be in the front plan of the robot.

    :param state: The current state of the robot (format: np.array([x, y, theta])).
    :param field_map: The FieldMap object. This is necessary to extract the true landmark positions in the field.
    :param max_observations: The maximum number of observations to generate per time step.
    :return: np.ndarray or size num_observations x 3. Each row is np.array([range, bearing, lm_id]).
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, FieldMap)

    assert state.shape == (3,)

    M = field_map.num_landmarks
    noise_free_observations_list = list()
    for k in range(M):
        noise_free_observations_list.append(get_observation(state, field_map, k))
    noise_free_observation_tuples = [(x[0], np.abs(x[1]), int(x[2])) for x in noise_free_observations_list]

    dtype = [('range', float), ('bearing', float), ('lm_id', int)]
    noise_free_observations = np.array(noise_free_observations_list)
    noise_free_observation_tuples = np.array(noise_free_observation_tuples, dtype=dtype)

    ii = np.argsort(noise_free_observation_tuples, order='bearing')
    noise_free_observations = noise_free_observations[ii]
    noise_free_observations[:, 2] = noise_free_observations[:, 2].astype(int)

    c1 = noise_free_observations[:, 1] > -np.pi / 2.
    c2 = noise_free_observations[:, 1] <  np.pi / 2.
    ii = np.nonzero((c1 & c2))[0]

    if ii.size <= max_observations:
        return noise_free_observations[ii]
    else:
        return noise_free_observations[:max_observations]


def generate_data(initial_pose,
                  num_steps,
                  num_landmarks_per_side,
                  max_obs_per_time_step,
                  alphas,
                  beta,
                  dt,
                  animate=False,
                  plot_pause_s=0.01):
    """
    Generates the trajectory of the robot using square path given by `generate_motion`.

    :param initial_pose: The initial pose of the robot in the field (format: np.array([x, y, theta])).
    :param num_steps: The number of time steps to generate the path for.
    :param num_landmarks_per_side: The number of landmarks to use on one side of the field.
    :param max_obs_per_time_step: The maximum number of observations to generate per time step of the sim.
    :param alphas: The noise parameters of the control actions (format: np.array([a1, a2, a3, a4])).
    :param beta: The noise parameter of observations (format: np.array([range (cm), bearing (deg)])).
    :param dt: The time difference (in seconds) between two consecutive time steps.
    :param animate: If True, this function will animate the generated data in a plot.
    :param plot_pause_s: The time (in seconds) to pause the plot animation between two consecutive frames.
    :return: SimulationData object.
    """

    # Initializations

    # State format: [x, y, theta]
    state_dim = 3
    # Motion format: [drot1, dtran, drot2]
    motion_dim = 3
    # Observation format: [range (cm, float),
    #                      bearing (rad, float),
    #                      landmark_id (id, int)]
    observation_dim = 3

    if animate:
        plt.figure(1)
        plt.ion()

    data_length = num_steps + 1
    filter_data = SlamInputData(np.zeros((data_length, motion_dim)),
                                np.empty((data_length, max_obs_per_time_step, observation_dim)))
    debug_data = SlamDebugData(np.zeros((data_length, state_dim)),
                               np.zeros((data_length, state_dim)),
                               np.empty((data_length, max_obs_per_time_step, observation_dim)))

    filter_data.observations[:] = np.nan
    debug_data.noise_free_observations[:] = np.nan

    debug_data.real_robot_path[0] = initial_pose
    debug_data.noise_free_robot_path[0] = initial_pose

    field_map = FieldMap(num_landmarks_per_side)

    # Covariance of observation noise.
    Q = np.diag([*(beta ** 2), 0])

    for i in range(1, data_length):
        # Simulate Motion

        # Noise-free robot motion command.
        t = i * dt
        filter_data.motion_commands[i] = generate_motion(t, dt)

        # Noise-free robot pose.
        debug_data.noise_free_robot_path[i] = \
            sample_from_odometry(debug_data.noise_free_robot_path[i - 1],
                                 filter_data.motion_commands[i],
                                 [0, 0, 0, 0])

        # Move the robot based on the noisy motion command execution.
        debug_data.real_robot_path[i] = sample_from_odometry(debug_data.real_robot_path[i - 1],
                                                             filter_data.motion_commands[i],
                                                             alphas)

        # Simulate Observation

        noise_free_observations = sense_landmarks(debug_data.real_robot_path[i], field_map, max_obs_per_time_step)
        noisy_observations = np.empty(noise_free_observations.shape)
        noisy_observations[:] = np.nan
        num_observations = noise_free_observations.shape[0]

        for k in range(num_observations):
            # Generate observation noise.
            observation_noise = sample2d(np.zeros(observation_dim), Q)
            # Generate noisy observation as observed by the robot for the filter.
            noisy_observations[k] = noise_free_observations[k] + observation_noise

        if noisy_observations.shape == (0, 3):
            print('hello')

        filter_data.observations[i] = noisy_observations
        debug_data.noise_free_observations[i] = noise_free_observations

        if animate:
            plt.clf()

            plot_field(field_map, noise_free_observations[:, 2])
            plot_robot(debug_data.real_robot_path[i])
            plot_observations(debug_data.real_robot_path[i],
                              debug_data.noise_free_observations[i],
                              filter_data.observations[i])

            plt.plot(debug_data.real_robot_path[1:i, 0], debug_data.real_robot_path[1:i, 1], 'b')
            plt.plot(debug_data.noise_free_robot_path[1:i, 0], debug_data.noise_free_robot_path[1:i, 1], 'g')

            plt.draw()
            plt.pause(plot_pause_s)

    if animate:
        plt.show(block=True)

    # This only initializes the sim data with everything but the first entry (which is just the prior for the sim).
    filter_data.motion_commands = filter_data.motion_commands[1:]
    filter_data.observations = filter_data.observations[1:]
    debug_data.real_robot_path = debug_data.real_robot_path[1:]
    debug_data.noise_free_robot_path = debug_data.noise_free_robot_path[1:]
    debug_data.noise_free_observations = debug_data.noise_free_observations[1:]

    return SimulationData(num_steps, filter_data, debug_data)


def save_data(data, file_path):
    """
    Saves the simulation's input data to the given filename.

    :param data: A tuple with the filter and debug data to save.
    :param file_path: The the full file path to which to save the data.
    """

    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'wb') as data_file:
        np.savez(data_file,
                 num_steps=data.num_steps,
                 noise_free_motion=data.filter.motion_commands,
                 real_observations=data.filter.observations,
                 noise_free_observations=data.debug.noise_free_observations,
                 real_robot_path=data.debug.real_robot_path,
                 noise_free_robot_path=data.debug.noise_free_robot_path)


def load_data(data_filename):
    """
    Load existing data from a given filename.
    Accepted file formats are pickled `npy` and MATLAB `mat` extensions.

    :param data_filename: The path to the file with the pre-generated data.
    :raises Exception if the file does not exist.
    :return: DataFile type.
    """

    if not os.path.isfile(data_filename):
        raise Exception('The data file {} does not exist'.format(data_filename))

    file_extension = data_filename[-3:]
    if file_extension not in {'mat', 'npy'}:
        raise TypeError('{} is an unrecognized file extension. Accepted file '
                        'formats include "npy" and "mat"'.format(file_extension))

    num_steps = 0
    filter_data = None
    debug_data = None

    if file_extension == 'npy':
        with np.load(data_filename) as data:
            num_steps = np.asscalar(data['num_steps'])
            filter_data = SlamInputData(data['noise_free_motion'], data['real_observations'])
            debug_data = SlamDebugData(data['real_robot_path'],
                                       data['noise_free_robot_path'],
                                       data['noise_free_observations'])
    elif file_extension == 'mat':
        data = scipy.io.loadmat(data_filename)
        if 'data' not in data:
            raise TypeError('Unrecognized data file')

        data = data['data']
        num_steps = data.shape[0]

        # Convert to zero-indexed landmark IDs.
        data[:, 1] -= 1
        data[:, 6] -= 1

        filter_data = SlamInputData(data[:, 2:5], data[:, 0:2])
        debug_data = SlamDebugData(data[:, 7:10], data[:, 10:13], data[:, 5:7])

    return SimulationData(num_steps, filter_data, debug_data)
