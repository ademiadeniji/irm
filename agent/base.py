class BaseAgent:
    """
    Base agent, shared across different algorithms, for a centralized
    observation processing framework
    """
    def __init__(self, obs_type):
        self.obs_type = obs_type
        self.discr_obs_dim = self.obs_dim 
        if self.obs_type in ["fetch_push_xy"]:
            self.discr_obs_dim = 2 
        elif self.obs_type in ["jaco_xyz", "walker_delta_xyz", "fetch_reach_xyz"]:
            self.discr_obs_dim = 3

    def get_reduced_observation(self, aug_obs):
        if self.obs_type == "jaco_xyz":
            proc_obs = aug_obs[:, 30:33]
        elif self.obs_type == "fetch_push_xy":
            proc_obs = aug_obs[:, 3:5]
        elif self.obs_type == "fetch_reach_xyz":
            proc_obs = aug_obs[:, :3]
        elif self.obs_type == "quad_velocity":
            proc_obs = aug_obs[:, 32:35]
        elif self.obs_type == "walker_delta_xyz":
            proc_obs = aug_obs[:, :3]
            aug_obs = aug_obs[:, 3:]
        else:
            proc_obs = aug_obs
        return proc_obs