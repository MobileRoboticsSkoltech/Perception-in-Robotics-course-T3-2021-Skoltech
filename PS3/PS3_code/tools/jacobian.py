"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
22-Feb-2020
"""

import numpy as np
from tools.task import get_motion_noise_covariance
from tools.task import wrap_angle
from scipy.linalg import cholesky


def state_jacobian(x, u):
    """
    Calculate jacobian of planar odometry in (x,u) point.

    :param x:
    :param u:
    :return:
    """
    J_x = np.eye(3)
    J_x[0, 2] = -u[1] * np.sin(x[2] + u[0])
    J_x[1, 2] = u[1] * np.cos(x[2] + u[0])


    # Note: This Jacobian is rank deficient for dtrans = 0
    # 
    if u[1] == 0:
        dtrans = 1e-4
    else:
        dtrans = u[1]
    J_u = np.array([[-dtrans * np.sin(x[2] + u[0]), np.cos(x[2] + u[0]), 0],
                    [dtrans * np.cos(x[2] + u[0]), np.sin(x[2] + u[0]), 0],
                    [1, 0, 1]])

    return J_x, J_u


def augmented_jacobian(x, u, n_landmarks):
    """
    Calculate augmented jacobian (with landmarks) of planar odometry transition function in (x,u) point.

    :param x: np.array([x,y,theta])
    :param u: np.array([delta_rot1,delta_trans,delta_rot2])
    :return: tuple(J_y,J_u)
    """
    J_x = np.eye(3 + 2 * n_landmarks)
    J_x[:3, :3], J_u = state_jacobian(x, u)

    J_u = np.vstack((J_u, np.zeros((2 * n_landmarks, 3))))
    return J_x, J_u


def observation_jacobian(x, m):
    """
    Calculate jacobian of planar odometry observation model in x point.
    :param x: np.array([x,y,theta])
    :param landmark: int, landmark index
    :param field_map: FieldMap object
    :return: tuple(J_x, J_m)
    """
    q = (m[0] - x[0]) ** 2 + (m[1] - x[1]) ** 2

    J_x = np.array([[-(m[0] - x[0]) / np.sqrt(q),
                     -(m[1] - x[1]) / np.sqrt(q), 0],
                    [(m[1] - x[1]) / q,
                     -(m[0] - x[0]) / q, -1]])
    J_m = -J_x[:2, :2]

    return J_x, J_m


def inverse_observation_jacobian(x, z):
    """
    Calculate jacobian of inverse observation function.

    :param x: np.array([x,y,theta])
    :param z: np.array([range, bearing])
    :return: tuple(J_x, J_z)
    """

    J_x = np.array([[1, 0, -z[0] * np.sin(z[1] + x[2])],
                    [0, 1, z[0] * np.cos(z[1] + x[2])]])

    J_z = np.array([[np.cos(z[1] + x[2]), -z[0] * np.sin(z[1] + x[2])],
                    [np.sin(z[1] + x[2]), z[0] * np.cos(z[1] + x[2])]])

    return J_x, J_z


def observation_augmented_jacobian(x, m, landmark_id, n_landmarks):
    """
    Calculate jacobian of observation.

    :param x: np.array([x,y(,delta)]), state;
    :param landmark: int, index of landmark;
    :param n_landmarks int, number of known landmaks;
    :return:
    """

    q = (m[0] - x[0]) ** 2 + (m[1] - x[1]) ** 2

    H_m = np.array([[(m[0] - x[0]) / np.sqrt(q), (m[1] - x[1]) / np.sqrt(q)],
                    [-(m[1] - x[1]) / q, (m[0] - x[0]) / q]])
    H = np.hstack((observation_jacobian(x, m)[0], np.zeros((2, 2 * landmark_id)), H_m,
                   np.zeros((2, 2 * (n_landmarks - landmark_id - 1)))))

    return H







    # def augmented_jacobians(n_landmarks, y_mu, u_mu, state_dim=3, landmark_dim=2, return_tuple=True):
    #     """
    #
    #     :param n_landmarks:
    #     :param y_mu: np.ndarray([x,y,theta,m1_x,m1_y,m2_x,m2_y,...]);
    #     :param u_mu: np.ndarray([delta_rot1, delta_trans, delta_rot2]);
    #     :param state_dim:
    #     :param return_tuple: whether to return serapate matrices for y and u;
    #     :return:
    #     """
    #
    #     J_x = np.eye(state_dim + n_landmarks * landmark_dim)
    #     J_x[0, 2] = -u_mu[1] * np.sin(y_mu[2] + u_mu[0])
    #     J_x[1, 2] = u_mu[1] * np.cos(y_mu[2] + u_mu[0])
    #
    #     J_u = np.eye(state_dim + n_landmarks * landmark_dim)
    #     V_t = np.eye(stat)
