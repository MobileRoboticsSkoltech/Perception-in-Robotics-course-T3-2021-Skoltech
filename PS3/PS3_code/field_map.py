"""
Sudhanva Sreesha
ssreesha@umich.edu
21-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018

Defines the field (a.k.a. map) for this task.
"""

import numpy as np


class FieldMap(object):
    def __init__(self, num_landmarks_per_side):
        """
        Initializes the map of the field.

        :param num_landmarks_per_side: The number of landmarks to use per side in the field.
        """

        self._complete_size_x = self._inner_size_x + 2 * self._inner_offset_x
        self._complete_size_y = self._inner_size_y + 2 * self._inner_offset_y
        self._num_landmarks_per_side = num_landmarks_per_side

        landmark_poses_x = self._landmark_offset_x + np.linspace(0, self._landmark_distance_x, num_landmarks_per_side)

        self._landmark_poses_x = np.hstack((landmark_poses_x, landmark_poses_x[::-1]))
        self._landmark_poses_y = self._landmark_offset_y + np.hstack((np.zeros(num_landmarks_per_side),
                                                                      np.full(num_landmarks_per_side,
                                                                              self._landmark_distance_y)))

    @property
    def _inner_offset_x(self):
        return 32

    @property
    def _inner_offset_y(self):
        return 13

    @property
    def _inner_size_x(self):
        return 420

    @property
    def _inner_size_y(self):
        return 270

    @property
    def _landmark_offset_x(self):
        return 21

    @property
    def _landmark_offset_y(self):
        return 0

    @property
    def _landmark_distance_x(self):
        return 442

    @property
    def _landmark_distance_y(self):
        return 292

    @property
    def num_landmarks(self):
        return self._num_landmarks_per_side * 2

    @property
    def complete_size_x(self):
        return self._complete_size_x

    @property
    def complete_size_y(self):
        return self._complete_size_y

    @property
    def landmarks_poses_x(self):
        return self._landmark_poses_x

    @property
    def landmarks_poses_y(self):
        return self._landmark_poses_y
