"""
Sudhanva Sreesha
ssreesha@umich.edu
24-Mar-2018

Gonzalo Ferrer,
g.ferrer@skoltech.ru

This file contains all utilities for plotting data.
"""

import numpy as np

from scipy.linalg import cholesky
from matplotlib import pyplot as plt

from field_map import FieldMap


def plot2dcov(mu, Sigma, color='k', nSigma=1, legend=None):
    """
    Plots a 2D covariance ellipse given the Gaussian distribution parameters.
    The function expects the mean and covariance matrix to ignore the theta parameter.

    :param mu: The mean of the distribution: 2x1 vector.
    :param Sigma: The covariance of the distribution: 2x2 matrix.
    :param color: The border color of the ellipse and of the major and minor axes.
    :param nSigma: The radius of the ellipse in terms of the number of standard deviations (default: 1).
    :param legend: If not None, a legend label to the ellipse will be added to the plot as an attribute.
    """
    mu = np.array(mu)
    assert mu.shape == (2,)
    Sigma = np.array(Sigma)
    assert Sigma.shape == (2, 2)

    n_points = 50

    A = cholesky(Sigma, lower=True)

    angles = np.linspace(0, 2 * np.pi, n_points)
    x_old = nSigma * np.cos(angles)
    y_old = nSigma * np.sin(angles)

    x_y_old = np.stack((x_old, y_old), 1)
    x_y_new = np.matmul(x_y_old, np.transpose(A)) + mu.reshape(1, 2) # (A*x)T = xT * AT

    plt.plot(x_y_new[:, 0], x_y_new[:, 1], color=color, label=legend)
    plt.scatter(mu[0], mu[1], color=color)


def plot_field(detected_marker):
    """
    Plots the field and highlights the currently detected marker.

    :param detected_marker: The marker id of the current detected marker.
    """

    margin = 200
    field_map = FieldMap()

    plt.axis((-margin, field_map.complete_size_x + margin, -margin, field_map.complete_size_y + margin))
    plt.xlabel('X')
    plt.ylabel('Y')

    for k in range(field_map.num_landmarks):
        center = [field_map.landmarks_poses_x[k], field_map.landmarks_poses_y[k]]

        if detected_marker == k:
            landmark = plt.Circle(center, 15, edgecolor='black', facecolor='gray')
        else:
            landmark = plt.Circle(center, 15, edgecolor='black', facecolor='none')

        plt.gcf().gca().add_artist(landmark)
        plt.text(center[0] - 2, center[1], str(k))


def plot_robot(state):
    """
    Plots a circle at the center of the robot and a line to depict the yaw.

    :param state: (x, y, theta)
    """

    assert isinstance(state, np.ndarray)
    assert state.shape == (3,)

    radius = 15
    robot = plt.Circle(state[:-1], radius, edgecolor='black', facecolor='cyan', alpha=0.25)
    orientation_line = np.array([[state[0], state[0] + (np.cos(state[2]) * (radius * 1.5))],
                                 [state[1], state[1] + (np.sin(state[2]) * (radius * 1.5))]])

    plt.gcf().gca().add_artist(robot)
    plt.plot(orientation_line[0], orientation_line[1], 'black')


def plot_observation(state, noise_free_observation, noisy_observation):
    """
    Plot two lines corresponding to the noisy and noise free observations from the robot to respective landmarks.

    :param state: The current robot pose: x, y, theta.
    :param noise_free_observation: Noise free bearing observation to the landmark (in rad).
    :param noisy_observation: Noisy bearing observation to the landmark (in rad).
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(noise_free_observation, np.ndarray)
    assert isinstance(noisy_observation, np.ndarray)

    assert state.shape == (3,)
    assert noise_free_observation.shape == (2,)
    assert noisy_observation.shape == (2,)

    # Plot the line to indicate observed landmarks (a.k.a. noisy observations).
    plt.plot([state[0], state[0] + 100 * np.cos(state[2] + noisy_observation[0])],
             [state[1], state[1] + 100 * np.sin(state[2] + noisy_observation[0])],
             'red')

    # Plot the line to indicate the true observations to landmarks (a.k.a. noise free observations).
    plt.plot([state[0], state[0] + 100 * np.cos(state[2] + noise_free_observation[0])],
             [state[1], state[1] + 100 * np.sin(state[2] + noise_free_observation[0])],
             'cyan')
