"""
Sudhanva Sreesha
ssreesha@umich.edu
24-Mar-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018

This file contains all utilities for plotting data.
"""

import numpy as np
from matplotlib import pyplot as plt



def plot2dcov(mu, Sigma, color, nSigma=1, legend=None):
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


def plot_robot(state, radius=15.):
    """
    Plots a circle at the center of the robot and a line to depict the yaw.

    :param state: numpy.ndarray([x, y, theta]).
    :param radius: The radius of the circle representing the robot.
    """

    assert isinstance(state, np.ndarray)
    assert state.shape == (3,)

    robot = plt.Circle(state[:-1], radius, edgecolor='black', facecolor='cyan', alpha=0.25)
    orientation_line = np.array([[state[0], state[0] + (np.cos(state[2]) * (radius * 1.5))],
                                 [state[1], state[1] + (np.sin(state[2]) * (radius * 1.5))]])

    plt.gcf().gca().add_artist(robot)
    plt.plot(orientation_line[0], orientation_line[1], 'black')




def get_plots_figure(should_show_plots, should_write_movie):
    """
    :param should_show_plots: Indicates whether the animation of SLAM should be plotted, in real time.
    :param should_write_movie: Indicates whether the animation of SLAM should be written to a movie file.
    :return: A figure if the plots should be shown or a movie file should be written, else None.
    """

    fig = None
    if should_show_plots or should_write_movie:
        fig = plt.figure(1)
    if should_show_plots:
        plt.ion()

    return fig


def plot_field(field_map, detected_landmarks):
    """
    Plots the field and highlights the currently detected marker.

    :param field_map: The FieldMap object to plot.
    :param detected_landmarks: 1d np.array with landmark indexes of all the detected landmarks at the current time step.
    """

    margin = 200

    plt.axis((-margin, field_map.complete_size_x + margin, -margin, field_map.complete_size_y + margin))
    plt.xlabel('X')
    plt.ylabel('Y')

    for k in range(field_map.num_landmarks):
        center = [field_map.landmarks_poses_x[k], field_map.landmarks_poses_y[k]]

        if k in detected_landmarks:
            landmark = plt.Circle(center, 15, edgecolor='black', facecolor='gray')
        else:
            landmark = plt.Circle(center, 15, edgecolor='black', facecolor='none')

        plt.gcf().gca().add_artist(landmark)
        plt.text(center[0] - 2, center[1], str(k))


def plot_observations(pose, noise_free_observations, noisy_observations):
    """
    Plot two lines corresponding to the noisy and noise free observations from the robot to respective landmarks.

    :param pose: The current robot pose: x, y, theta.
    :param noise_free_observations: A 2-d np.ndarray of noise free observations (size: Mx3) of all detected landmarks.
    :param noisy_observations: A 2-d np.ndarray of noisy observations (size: Mx3) of all the detected landmarks.
    """

    assert isinstance(noise_free_observations, np.ndarray)
    assert isinstance(noisy_observations, np.ndarray)

    assert noise_free_observations.shape == noisy_observations.shape

    M = noise_free_observations.shape[0]
    for k in range(M):
        noisy_range, noisy_bearing, _ = noisy_observations[k]
        nf_range, nf_bearing, _ = noise_free_observations[k]

        # Plot the line to indicate observed landmarks (a.k.a. noisy observations).
        plt.plot([pose[0], pose[0] + noisy_range * np.cos(pose[2] + noisy_bearing)],
                 [pose[1], pose[1] + noisy_range * np.sin(pose[2] + noisy_bearing)],
                 'brown')

        # Plot the line to indicate the true observations to landmarks (a.k.a. noise free observations).
        plt.plot([pose[0], pose[0] + nf_range * np.cos(pose[2] + nf_bearing)],
                 [pose[1], pose[1] + nf_range * np.sin(pose[2] + nf_bearing)],
                 'cyan')
