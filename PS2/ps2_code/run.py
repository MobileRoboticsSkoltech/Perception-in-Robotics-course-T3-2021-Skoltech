#!/usr/bin/python

"""
Sudhanva Sreesha,
ssreesha@umich.edu,
28-Mar-2018

Gonzalo Ferrer,
g.ferrer@skoltech.ru
"""

import contextlib
import os
from argparse import ArgumentParser
from warnings import warn

import numpy as np
from matplotlib import animation as anim
from matplotlib import pyplot as plt
from progress.bar import FillingCirclesBar

from filters.ekf import EKF
from filters.pf import PF
from tools.data import generate_data as generate_input_data
from tools.data import load_data
from tools.data import save_data as save_input_data
from tools.objects import FilterTrajectory
from tools.objects import Gaussian
from tools.plot import plot2dcov
from tools.plot import plot_field
from tools.plot import plot_observation
from tools.plot import plot_robot


@contextlib.contextmanager
def get_dummy_context_mgr():
    """
    :return: A dummy context manager for conditionally writing to a movie file.
    """
    yield None


def get_cli_args():
    parser = ArgumentParser('Runs localization filters (EKF or PF) on generated, simulation data.')
    parser.add_argument('-i',
                        '--input-data-file',
                        type=str,
                        action='store',
                        help='File with generated data to simulate the filter '
                             'against. Supported format: "npy", and "mat".')
    parser.add_argument('-n',
                        '--num-steps',
                        type=int,
                        action='store',
                        help='The number of time steps to generate data for the simulation. '
                             'This option overrides the data file argument.',
                        default=100)
    parser.add_argument('-f',
                        '--filter',
                        dest='filter_name',
                        choices=['ekf', 'pf'],
                        action='store',
                        help='The localization filter to use for the simulation.',
                        default='ekf')
    parser.add_argument('--num-particles',
                        type=int,
                        action='store',
                        help='The number of particles to use in the PF.',
                        default=100)
    parser.add_argument('--global-localization',
                        action='store_true',
                        help='Uniformly distributes the particles around the field at the beginning of the simulation.')
    parser.add_argument('-a',
                        '--alphas',
                        nargs=4,
                        metavar=('A1', 'A2', 'A3', 'A4'),
                        action='store',
                        help='Squared root of alphas, used for transition noise in action space (M_t). (format: a1 a2 a3 a4).',
                        default=(0.05, 0.001, 0.05, 0.01))
    parser.add_argument('-b',
                        '--beta',
                        type=float,
                        action='store',
                        help='Diagonal of Standard deviations of the Observation noise Q. (format: deg).',
                        default=20)
    parser.add_argument('--dt', type=float, action='store', help='Time step (in seconds).', default=0.1)
    # TODO remove animate and let just -s
    parser.add_argument('--animate', action='store_true', help='Show and animation of the simulation, in real-time.')
    parser.add_argument('--show-particles',
                        action='store_true',
                        help='Show the particles when using the particle filter.')
    parser.add_argument('-s', '--show-trajectory',
                        action='store_true',
                        help='Shows the full robot trajectory as estimated by the filter. '
                             'If --show-particles is also specified with the particle filter, '
                             'this option will show one trajectory per particle (warn: this'
                             'can be chaotic to look at).')
    parser.add_argument('--plot-pause-len',
                        type=float,
                        action='store',
                        help='Time (in seconds) to pause the plot animation for between frames.',
                        default=0.01)
    parser.add_argument('-m',
                        '--movie-file',
                        type=str,
                        help='The full path to movie file to write the simulation animation to.',
                        default=None)
    parser.add_argument('--movie-fps',
                        type=float,
                        action='store',
                        help='The FPS rate of the movie to write.',
                        default=10.)
    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        default=None,
                        action='store',
                        help='The output directory to which the input and '
                             'output data from the simulation will be stored.')
    parser.add_argument('--global_localization', action='store_true', help='Task E, Global localization enabled, only for PF.')
    return parser.parse_args()


def validate_cli_args(args):
    if args.input_data_file and not os.path.exists(args.input_data_file):
        raise OSError('The input data file {} does not exist.'.format(args.input_data_file))

    if not args.input_data_file and not args.num_steps:
        raise RuntimeError('Neither `--input-data-file` nor `--num-steps` were present in the arguments.')

    if args.filter_name != 'pf' and args.global_localization:
        warn('Global localization is only supported for the particle filter. Ignoring the flag.')

    if not args.animate:
        if args.show_trajectory:
            warn('Since animation for the simulation was not enabled, ignoring `--show-trajectory`.')
        if args.show_particles:
            warn('Since animation for the simulation was not enabled, ignoring `--show-particles`.')

    if args.show_particles and args.filter_name != 'pf':
        warn('Since the simulation is not running the particle filter, ignoring `--show-particles`.')

    if args.show_particles and args.output_dir:
        warn('Since `--output-dir` is specified, ignoring `--show-particles` to generate just one trajectory.')


def main():
    args = get_cli_args()
    validate_cli_args(args)

    # weights for covariance action noise R and observation noise Q
    alphas = np.array(args.alphas) **2 # variance of noise R proportional to alphas, see tools/tasks@get_motion_noise_covariance()
    beta = np.deg2rad(args.beta) # see also filters/localization_filter.py

    mean_prior = np.array([180., 50., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    if args.input_data_file:
        data = load_data(args.input_data_file)
    elif args.num_steps:
        # Generate data, assuming `--num-steps` was present in the CL args.
        data = generate_input_data(initial_state.mu.T, args.num_steps, alphas, beta, args.dt)
    else:
        raise RuntimeError('')

    store_sim_data = True if args.output_dir else False
    show_plots = True if args.animate else False
    write_movie = True if args.movie_file else False
    show_trajectory = True if args.animate and args.show_trajectory else False
    show_particles = args.show_particles and args.animate and args.filter_name == 'pf'
    update_mean_trajectory = True if show_trajectory or store_sim_data else False
    update_plots = True if show_plots or write_movie else False
    one_trajectory_per_particle = True if show_particles and not store_sim_data else False

    if store_sim_data:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_input_data(data, os.path.join(args.output_dir, 'input_data.npy'))

    # ---------------------------------------------------------------------------------------------------
    # Student's task: You will fill these function inside 'filters/.py'
    # ---------------------------------------------------------------------------------------------------
    localization_filter = None
    if args.filter_name == 'ekf':
        localization_filter = EKF(initial_state, alphas, beta)
    elif args.filter_name == 'pf':
        localization_filter = PF(initial_state, alphas, beta, args.num_particles, args.global_localization)

    fig = None
    if show_plots or write_movie:
        fig = plt.figure(1)
    if show_plots:
        plt.ion()

    # Initialize the trajectory if user opted-in to display.
    sim_trajectory = None
    if update_mean_trajectory:
        if one_trajectory_per_particle:
            mean_trajectory = np.zeros((data.num_steps, localization_filter.state_dim, args.num_particles))
        else:
            mean_trajectory = np.zeros((data.num_steps, localization_filter.state_dim))

        sim_trajectory = FilterTrajectory(mean_trajectory)

    if store_sim_data:
        # Pre-allocate the memory to store the covariance matrix of the trajectory at each time step.
        sim_trajectory.covariance = np.zeros((localization_filter.state_dim,
                                              localization_filter.state_dim,
                                              data.num_steps))

    # Initialize the movie writer if `--movie-file` was present in the CL args.
    movie_writer = None
    if write_movie:
        get_ff_mpeg_writer = anim.writers['ffmpeg']
        metadata = dict(title='Localization Filter', artist='matplotlib', comment='PS2')
        movie_fps = min(args.movie_fps, float(1. / args.plot_pause_len))
        movie_writer = get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)

    progress_bar = FillingCirclesBar('Simulation Progress', max=data.num_steps)

    with movie_writer.saving(fig, args.movie_file, data.num_steps) if write_movie else get_dummy_context_mgr():
        for t in range(data.num_steps):
            # Used as means to include the t-th time-step while plotting.
            tp1 = t + 1

            # Control at the current step.
            u = data.filter.motion_commands[t]
            # Observation at the current step.
            z = data.filter.observations[t]

            localization_filter.predict(u)
            localization_filter.update(z)

            if update_mean_trajectory:
                if one_trajectory_per_particle:
                    sim_trajectory.mean[t, :, :] = localization_filter.X.T
                else:
                    sim_trajectory.mean[t] = localization_filter.mu

            if store_sim_data:
                sim_trajectory.covariance[:, :, t] = localization_filter.Sigma

            progress_bar.next()

            if not update_plots:
                continue

            plt.cla()
            plot_field(z[1])
            plot_robot(data.debug.real_robot_path[t])
            plot_observation(data.debug.real_robot_path[t],
                             data.debug.noise_free_observations[t],
                             data.filter.observations[t])

            plt.plot(data.debug.real_robot_path[1:tp1, 0], data.debug.real_robot_path[1:tp1, 1], 'g')
            plt.plot(data.debug.noise_free_robot_path[1:tp1, 0], data.debug.noise_free_robot_path[1:tp1, 1], 'm')

            #plt.plot([data.debug.real_robot_path[t, 0]], [data.debug.real_robot_path[t, 1]], '*g')
            plt.plot([data.debug.noise_free_robot_path[t, 0]], [data.debug.noise_free_robot_path[t, 1]], '*m')

            if show_particles:
                samples = localization_filter.X.T
                plt.scatter(samples[0], samples[1], s=2)
            else:
                plot2dcov(localization_filter.mu_bar[:-1],
                          localization_filter.Sigma_bar[:-1, :-1],
                          'red', 3,
                          legend='{} -'.format(args.filter_name.upper()))
                plot2dcov(localization_filter.mu[:-1],
                          localization_filter.Sigma[:-1, :-1],
                          'blue', 3,
                          legend='{} +'.format(args.filter_name.upper()))
                plt.legend()

            if show_trajectory:
                if len(sim_trajectory.mean.shape) > 2:
                    # This means that we probably intend to show the trajectory for ever particle.
                    x = np.squeeze(sim_trajectory.mean[0:t, 0, :])
                    y = np.squeeze(sim_trajectory.mean[0:t, 1, :])
                    plt.plot(x, y)
                else:
                    plt.plot(sim_trajectory.mean[0:t, 0], sim_trajectory.mean[0:t, 1], 'blue')

            if show_plots:
                # Draw all the plots and pause to create an animation effect.
                plt.draw()
                plt.pause(args.plot_pause_len)

            if write_movie:
                movie_writer.grab_frame()

    progress_bar.finish()

    if show_plots:
        plt.show(block=True)

    if store_sim_data:
        file_path = os.path.join(args.output_dir, 'output_data.npy')
        with open(file_path, 'wb') as data_file:
            np.savez(data_file,
                     mean_trajectory=sim_trajectory.mean,
                     covariance_trajectory=sim_trajectory.covariance)


if __name__ == '__main__':
    main()
