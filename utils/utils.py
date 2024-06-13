import random
import numpy as np
import torch


def reproducibility_settings(seed: int = 0):
    r"""Set the random seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # Settings for 3090
    torch.backends.cudnn.allow_tf32 = False  # Settings for 3090


def round_floats(o):
        if isinstance(o, float):
            return round(o, 6)
        if isinstance(o, dict):
            return {k: round_floats(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [round_floats(x) for x in o]
        return o


def augment_trajectory(obs_traj, pred_traj, flip=True, reverse=True):
    r"""Flip and reverse the trajectory

    Args:
        obs_traj (torch.Tensor): observed trajectory with shape (num_peds, obs_len, 2)
        pred_traj (torch.Tensor): predicted trajectory with shape (num_peds, pred_len, 2)
        flip (bool): whether to flip the trajectory
        reverse (bool): whether to reverse the trajectory
    """

    if flip:
        obs_traj = torch.cat([obs_traj, obs_traj * torch.FloatTensor([[[1, -1]]])], dim=0)
        pred_traj = torch.cat([pred_traj, pred_traj * torch.FloatTensor([[[1, -1]]])], dim=0)
    elif reverse:
        full_traj = torch.cat([obs_traj, pred_traj], dim=1)  # NTC
        obs_traj = torch.cat([obs_traj, full_traj.flip(1)[:, :obs_traj.size(1)]], dim=0)
        pred_traj = torch.cat([pred_traj, full_traj.flip(1)[:, obs_traj.size(1):]], dim=0)
    return obs_traj, pred_traj


if __name__ == '__main__':
    pass
