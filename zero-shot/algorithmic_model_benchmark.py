import argparse
import os
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import simdkalman
from utils.trajectory_dataset import get_dataloader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def KalmanFilter(obs_frames, obs_traj):
    r"""Kalman Filter
    Args:
        obs_frames (np.ndarray): observed frames
        obs_traj (np.ndarray): observed trajectory
    Returns:
        pred_traj (np.ndarray): predicted trajectory
    """
    kf = simdkalman.KalmanFilter(
        state_transition=np.array([[1, 1], [0, 1]]),
        process_noise=np.diag([0.01, 0.01]),
        observation_model=np.array([[1, 0]]),
        observation_noise=1e-3)
    
    # predict new data
    # results = kf.compute(obs_traj.transpose(1, 0), 12)
    predicted = kf.predict(obs_traj.transpose(1, 0), 12)
    
    return predicted.observations.mean.transpose(1, 0)


def eval_method(args):
    data_set = './datasets/' + args.dataset + '/'
    print('Scene: {}'.format(args.dataset))

    args.batch_size = 1e8
    loader_test = get_dataloader(data_set, 'test', args.obs_len, args.pred_len, args.batch_size)

    # Preprocessing
    obs_traj = loader_test.dataset.obs_traj
    pred_traj = loader_test.dataset.pred_traj

    n_ped, t_obs, dim = obs_traj.shape
    n_ped, t_pred, dim = pred_traj.shape

    # linear extrapolation
    obs_frames = np.arange(t_obs)
    pred_frames = np.arange(t_obs, t_obs + t_pred)
    extrp_traj_linear = np.zeros((n_ped, t_pred, dim))
    for i in range(n_ped):
        for j in range(dim):
            if args.model == 'stop':
                model_name = 'Fill Last Value'
                f = interp1d(obs_frames, obs_traj[i, :, j], bounds_error=False, kind='linear', fill_value=(obs_traj[i, 0, j], obs_traj[i, -1, j]))
                extrp_traj_linear[i, :, j] = f(pred_frames)
            elif args.model == 'linear':
                model_name = 'Linear Extrapolation'
                f = interp1d(obs_frames, obs_traj[i, :, j], fill_value='extrapolate')
                extrp_traj_linear[i, :, j] = f(pred_frames)
            elif args.model == 'cubic':
                model_name = 'Cubic Spline'
                f = CubicSpline(obs_frames, obs_traj[i, :, j], bc_type='natural')
                extrp_traj_linear[i, :, j] = f(pred_frames)
            elif args.model == 'kalman':
                break
            else:
                raise NotImplementedError
        if args.model == 'kalman':
            model_name = 'Kalman Filter'
            p = KalmanFilter(obs_frames, obs_traj[i])
            extrp_traj_linear[i] = p

    error_temp = np.linalg.norm(extrp_traj_linear - pred_traj, ord=2, axis=-1)
    print('model: {}'.format(model_name), end='\t')
    print('ADE: {:.4f}'.format(error_temp.mean(axis=-1).mean()), end='\t')
    print('FDE: {:.4f}'.format(error_temp[:, -1].mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='kalman', type=str, help="model for extrapolation")
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    args = parser.parse_args()

    dataset_all = ["eth", "hotel", "univ", "zara1", "zara2"]

    for scene in dataset_all:
        args.dataset = scene
        eval_method(args)
