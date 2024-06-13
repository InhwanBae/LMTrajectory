from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from utils.homography import world2image


def postprocess_trajectory(traj, obs_traj, seq_start_end, scene_id, homography, scene_map, cfg):
    # postprocess the trajectory
    def get_value(S, i, j):
        try:
            return S[i, j]
        except IndexError:
            return 0

    if cfg.deterministic:
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess")):
            map_temp = scene_map[scene_id[s]]
            # map_temp = np.ones_like(map_temp)  # Simply clipping into the scene
            for ped_id in range(s, e):
                sample = 0
                endpoint = (traj[ped_id, sample, -1] / cfg.IMAGE_SCALE_DOWN).astype(np.int32)
                if get_value(map_temp, endpoint[1], endpoint[0]) != 1:
                    # Pedestrian is in the wall, scale down.
                    obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                    # startpoint = traj[ped_id, sample, 0].copy()
                    startpoint = obs_traj_temp[-1].copy()

                    if cfg.dataset_name not in ["eth", "hotel", "univ", "zara1", 'zara2']:
                        if get_value(map_temp, startpoint[1], startpoint[0]) == 0:
                            # Pedestrian is already in the wall, skip.
                            continue

                    # print(traj[ped_id, sample, 0].copy(), obs_traj_temp[-1].copy())
                    # print(obs_traj_temp.shape, startpoint.shape)
                    scale = np.linspace(1.0, 0.01, 100)
                    traj_temp = traj[ped_id, sample].copy() - startpoint
                    traj_temps = np.tile(traj_temp[None, :, :], [len(scale), 1, 1])
                    traj_temps *= np.tile(scale[:, None, None], [1, *traj[ped_id, sample].shape])
                    traj_temps += startpoint
                    endpoints = (traj_temps[:, -1] / cfg.IMAGE_SCALE_DOWN).astype(np.int32)

                    # Find the first endpoint that is not in the wall
                    for i in range(len(endpoints)):
                        if get_value(map_temp, endpoints[i, 1], endpoints[i, 0]) == 1:
                            break
                    traj[ped_id, sample] = traj_temps[i]

    else:
        new_traj = np.zeros([traj.shape[0], cfg.best_of_n, cfg.pred_len, 2])
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess")):
            map_temp = scene_map[scene_id[s]]
            # if simple clipping into the scene,
            # map_temp = np.ones_like(map_temp)
            for ped_id in range(s, e):
                obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                startpoint = obs_traj_temp[-1].copy()

                # Sample removal if there are abnormal movements
                THRESHOLD = 100
                mask = np.diff(traj[ped_id, :, :, :], n=1, axis=1)
                mask = np.linalg.norm(mask, ord=2, axis=-1)
                mask = np.any(np.greater(mask, THRESHOLD), axis=1)
                # traj_filtered = traj[ped_id, ~mask]
                traj_filtered = traj[ped_id].copy()
                traj_filtered[mask, :, 0] = startpoint[0]
                traj_filtered[mask, :, 1] = startpoint[1]
                max_samples_filtered = traj_filtered.shape[0]

                obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                startpoint = obs_traj_temp[-1].copy()

                for sample in range(max_samples_filtered):
                    endpoint = (traj_filtered[sample, -1] / cfg.IMAGE_SCALE_DOWN).astype(np.int32)
                    if get_value(map_temp, endpoint[1], endpoint[0]) != 1:
                        # Pedestrian is in the wall, scale down.

                        if cfg.dataset_name not in ["eth", "hotel", "univ", "zara1", 'zara2']:
                            if get_value(map_temp, startpoint[1], startpoint[0]) == 0:
                                # Pedestrian is already in the wall, skip.
                                continue

                        scale = np.linspace(1.0, 0.01, 100)
                        traj_temp = traj_filtered[sample].copy() - startpoint
                        traj_temps = np.tile(traj_temp[None, :, :], [len(scale), 1, 1])
                        traj_temps *= np.tile(scale[:, None, None], [1, cfg.pred_len, 2])
                        traj_temps += startpoint
                        endpoints = (traj_temps[:, -1] / cfg.IMAGE_SCALE_DOWN).astype(np.int32)

                        # Find the first endpoint that is not in the wall
                        for i in range(len(endpoints)):
                            if get_value(map_temp, endpoints[i, 1], endpoints[i, 0]) == 1:
                                break
                        traj_filtered[sample] = traj_temps[i]

                # Clustering
                if max_samples_filtered > cfg.best_of_n:
                    temp = traj_filtered.reshape(max_samples_filtered, -1)
                    centroids = KMeans(n_clusters=cfg.best_of_n, random_state=0, init='k-means++', n_init=1).fit(temp).cluster_centers_
                    traj_filtered = centroids.reshape(cfg.best_of_n, cfg.pred_len, -1)

                new_traj[ped_id, :traj_filtered.shape[0]] = traj_filtered

        traj = new_traj
    return traj


def postprocess_trajectory_simple(traj, obs_traj, seq_start_end, scene_id, homography, cfg):
    if cfg.deterministic:
        pass
    else:
        new_traj = np.zeros([traj.shape[0], cfg.best_of_n, cfg.pred_len, 2])
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess")):
            for ped_id in range(s, e):
                if cfg.metric == "pixel":
                    startpoint = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])[-1].copy()
                else:
                    startpoint = obs_traj[ped_id][-1].copy()

                # Sample removal if there are abnormal movements
                THRESHOLD = 100 if cfg.metric == "pixel" else 5
                mask = np.diff(traj[ped_id, :, :, :], n=1, axis=1)
                mask = np.linalg.norm(mask, ord=2, axis=-1)
                mask = np.any(np.greater(mask, THRESHOLD), axis=1)
                # traj_filtered = traj[ped_id, ~mask]
                traj_filtered = traj[ped_id].copy()
                traj_filtered[mask, :, 0] = startpoint[0]
                traj_filtered[mask, :, 1] = startpoint[1]
                max_samples_filtered = traj_filtered.shape[0]

                # Clustering
                if max_samples_filtered > cfg.best_of_n:
                    temp = traj_filtered.reshape(max_samples_filtered, -1)
                    centroids = KMeans(n_clusters=cfg.best_of_n, random_state=0, init='k-means++', n_init=1).fit(
                        temp).cluster_centers_
                    traj_filtered = centroids.reshape(cfg.best_of_n, cfg.pred_len, -1)

                new_traj[ped_id, :traj_filtered.shape[0]] = traj_filtered

        traj = new_traj
    return traj
