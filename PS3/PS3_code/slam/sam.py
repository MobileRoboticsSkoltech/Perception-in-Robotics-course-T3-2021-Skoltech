"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance

class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas
        
        self.graph = mrob.FGraph()
        

    def predict(self, u):
        pass

    def update(self, z):
        pass
