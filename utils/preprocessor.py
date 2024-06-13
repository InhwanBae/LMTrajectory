import os
import json
import copy
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import get_dataloader
from converter import batch_traj2txt, change_template
from utils import reproducibility_settings, round_floats
from homography import world2image, image2world, generate_homography

hist_ped_template = "Pedestrian {} moved along the trajectory {} for {} frames."
question_answer_template = "question: {} context: {} answer:"
question_template = {"forecast": "What trajectory does pedestrian {} follow for the next {} frames?",
                     "destination": "At which coordinates does pedestrian {} arrive after the next {} frames?",
                     "direction": "In which direction will pedestrian {} move in the future?",
                     "group": "With which pedestrians does pedestrian {} form a group?",
                     "collision": "With which pedestrian does pedestrian {} have a collision risk?",
                     "mimicry": "Which pedestrian seems to walk similarly to pedestrian {}?"}
answer_template = {"forecast": "Pedestrian {} will move along the trajectory {} for the next {} frames.",
                   "destination": "Pedestrian {} will arrive at coordinate {} after the next {} frames.",
                   "direction": "Pedestrian {} will {}.",
                   "group": "Pedestrian {} forms a group with pedestrian {}.",
                   "collision": "Pedestrian {} has a collision risk with pedestrian {}.",
                   "mimicry": "Pedestrian {} walks similarly to pedestrian {}.",
                   "direction_answer_list": ["move forward", "move backward", "move left", "move right", "stop"],
                   "group_false": "Pedestrian {} will walk alone.",
                   "collision_false": "Pedestrian {} has no collision risk.",
                   "mimicry_false": "Pedestrian {} will walk alone."}

IMAGE_SCALE_DOWN = 0.25
AUGMENTATION_SHIFT = [-2, -1, 0, 1, 2]
NUMPED_THRESHOLD = 5
COORD_PRECISION = {"meter": "{:.2f}", "pixel": "{:d}"}


def get_direction_type(obs_traj, pred_traj):
    DIRECTION_RADIUS_THRESHOLD = 0.2
    DIRECTION_BACKWARD_THRESHOLD = 90
    DIRECTION_FORWARD_THRESHOLD = 30
    obs_len, pred_len = obs_traj.shape[1], pred_traj.shape[1]
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)[0]
    full_traj_disp = full_traj[1:] - full_traj[:-1]
    # Filter out static people
    if np.linalg.norm(full_traj[obs_len] - full_traj[pred_len], ord=2, axis=-1) < DIRECTION_RADIUS_THRESHOLD:
        return 4
    # Normalize rotation
    dir = full_traj[obs_len] - full_traj[obs_len - 2]
    rot = np.arctan2(dir[1], dir[0])
    traj_rot = np.array([[np.cos(rot), np.sin(-rot)],
                         [np.sin(rot), np.cos(rot)]])
    full_traj_disp_norm = full_traj_disp @ traj_rot
    future_norm = full_traj_disp_norm[obs_len:].mean(axis=0)
    future_dir = np.arctan2(future_norm[1], future_norm[0]) * 180 / np.pi
    # Filter out moving backward people
    if future_dir < -DIRECTION_BACKWARD_THRESHOLD or future_dir > DIRECTION_BACKWARD_THRESHOLD:
        return 1
    # Filter out moving left people
    if future_dir > DIRECTION_FORWARD_THRESHOLD:
        return 2
    # Filter out moving right people
    if future_dir < -DIRECTION_FORWARD_THRESHOLD:
        return 3
    # Moving forward
    return 0


def get_group_id(obs_traj, pred_traj):
    GROUP_DIST_THRESHOLD = 1
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)
    full_traj_target = full_traj[[0]]
    full_traj_neighbor = full_traj[1:]
    distance = np.linalg.norm(full_traj_neighbor - full_traj_target, ord=2, axis=-1)
    mask = np.all(distance < GROUP_DIST_THRESHOLD, axis=-1)
    nearest = distance.mean(axis=-1).argsort()
    for id in nearest:
        if mask[id]:
            return id + 1
    return None


def get_collision_id(obs_traj, pred_traj):
    COLLISION_DIST_THRESHOLD = 0.3
    full_traj = pred_traj
    full_traj_target = full_traj[[0]]
    full_traj_neighbor = full_traj[1:]
    distance = np.linalg.norm(full_traj_neighbor - full_traj_target, ord=2, axis=-1)
    mask = np.any(distance < COLLISION_DIST_THRESHOLD, axis=-1)
    nearest = distance.min(axis=-1).argsort()
    for id in nearest:
        if mask[id]:
            return id + 1
    return None


def get_mimicry_id(obs_traj, pred_traj):
    MIMICRY_DISP_THRESHOLD = 0.1
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)
    full_traj_disp = full_traj[:, 1:] - full_traj[:, :-1]
    full_traj_disp_target = full_traj_disp[[0]]
    full_traj_disp_neighbor = full_traj_disp[1:]
    distance = np.linalg.norm(full_traj_disp_neighbor - full_traj_disp_target, ord=2, axis=-1)
    mask = distance.mean(axis=-1) < MIMICRY_DISP_THRESHOLD
    nearest = distance.mean(axis=-1).argsort()
    for id in nearest:
        if mask[id]:
            return id + 1
    return None


def preprocess_dataset(dataset, phase, obs_len, pred_len, coord_system="meter", use_scene_context=True, 
                       multimodal=["forecast", "destination", "direction", "group", "collision", "mimicry"], 
                       augment=["shuffle", "shift", "flip", "swap", "reverse"],
                       postfix="-multimodal"):
    data_dir = f"./datasets/{dataset}"
    dst_file = f"./datasets/preprocessed/{dataset}-{phase}-{obs_len}-{pred_len}-{coord_system}{postfix}.json"
   
    if phase != "train":
        augment = []

    os.makedirs("./datasets/preprocessed/", exist_ok=True)

    reproducibility_settings()
    
    dataloader = get_dataloader(data_dir, phase, obs_len, pred_len, batch_size=1)
    
    homography = dataloader.dataset.homography
    scene_img = dataloader.dataset.scene_img
    scene_map = dataloader.dataset.scene_map
    scene_desc = dataloader.dataset.scene_desc

    # Change coordinate precision
    if coord_system == "meter":
        change_template({"coord_template": COORD_PRECISION[coord_system]})
    elif coord_system == "pixel":
        change_template({"coord_template": COORD_PRECISION[coord_system]})
        # Scale down the scene
        for k, v in homography.items():
            homography[k] = v.copy() @ generate_homography(scale=IMAGE_SCALE_DOWN)
    else:
        raise NotImplementedError
    
    processed_data = {m: [] for m in multimodal}

    for batch in tqdm(dataloader, desc=f"Processing {dataset} {phase} dataset..."):
        obs_traj = batch['obs_traj'].numpy()
        pred_traj = batch['pred_traj'].numpy()
        non_linear_ped = batch['non_linear_ped'].numpy()
        loss_mask = batch['loss_mask'].numpy()
        scene_mask = batch['scene_mask'].numpy()
        seq_start_end = batch['seq_start_end'].numpy()
        frame = batch['frame'].numpy()
        scene_id = batch['scene_id']
        n_ped = obs_traj.shape[0]

        aug_param = []
        if "shift" in augment:
            for x_shift in AUGMENTATION_SHIFT:
                for y_shift in AUGMENTATION_SHIFT:
                    aug_param.append({"shift": np.array([x_shift, y_shift]), "flip": False, "swap": False, "reverse": False})
        else:
            aug_param.append({"shift": np.array([0, 0]), "flip": False, "swap": False, "reverse": False})

        if "flip" in augment:
            aug_param_temp = copy.deepcopy(aug_param)
            for ap in aug_param_temp:
                ap["flip"] = True
            aug_param.extend(aug_param_temp)

        if "swap" in augment:
            aug_param_temp = copy.deepcopy(aug_param)
            for ap in aug_param_temp:
                ap["swap"] = True
            aug_param.extend(aug_param_temp)
        
        if "reverse" in augment:
            aug_param_temp = copy.deepcopy(aug_param)
            for ap in aug_param_temp:
                ap["reverse"] = True
            aug_param.extend(aug_param_temp)

        # Generate questions and answers for each pedestrian.
        for ped_id in range(n_ped):
            obs_traj_trunc = obs_traj.copy()
            pred_traj_trunc = pred_traj.copy()

            # Dropout the pedestrian if there are too many pedestrians in the scene.
            obs_traj_target = obs_traj_trunc[[ped_id]]
            obs_nearest = np.linalg.norm(obs_traj_trunc - obs_traj_target, ord=2, axis=-1)[:, -1].argsort()
            obs_dropout = obs_nearest[1:NUMPED_THRESHOLD]
            if "shuffle" in augment:
                np.random.shuffle(obs_dropout[1:])
            obs_dropout = np.concatenate([np.array([ped_id]), obs_dropout])
            obs_traj_trunc = obs_traj_trunc[obs_dropout]
            pred_traj_trunc = pred_traj_trunc[obs_dropout]
            n_ped_temp = obs_traj_trunc.shape[0]

            # Calculate multimodal answers
            direction_type = get_direction_type(obs_traj_trunc, pred_traj_trunc) if "direction" in multimodal else None
            group_id = get_group_id(obs_traj_trunc, pred_traj_trunc) if "group" in multimodal else None
            collision_id = get_collision_id(obs_traj_trunc, pred_traj_trunc) if "collision" in multimodal else None
            mimicry_id = get_mimicry_id(obs_traj_trunc, pred_traj_trunc) if "mimicry" in multimodal else None

            # Pixel scale transformation
            if coord_system == "pixel":
                H = homography[scene_id[ped_id]]
                obs_traj_trunc = world2image(obs_traj_trunc, H).astype(np.int32)
                pred_traj_trunc = world2image(pred_traj_trunc, H).astype(np.int32)

            # Augment the trajectory.
            for ap in aug_param:
                obs_traj_aug = obs_traj_trunc.copy()
                pred_traj_aug = pred_traj_trunc.copy()

                obs_traj_aug += ap["shift"][None, None, :]
                pred_traj_aug += ap["shift"][None, None, :]
                if ap["flip"]:
                    if coord_system == "pixel":
                        scene_img_size = (np.array(scene_img[scene_id[ped_id]].size)[None, None, :] * IMAGE_SCALE_DOWN).astype(np.int32)
                    elif coord_system == "meter":
                        mid = [3.0, 5.0] if scene_id[ped_id] == "biwi_eth" else [2.5, -3.0] if scene_id[ped_id] == "biwi_hotel" else [7.5, 7.5]
                        scene_img_size = (np.array(mid)[None, None, :])
                    obs_traj_aug = -obs_traj_aug + scene_img_size
                    pred_traj_aug = -pred_traj_aug + scene_img_size
                if ap["swap"]:
                    obs_traj_aug = obs_traj_aug[:, :, [1, 0]]
                    pred_traj_aug = pred_traj_aug[:, :, [1, 0]]
                if ap["reverse"]:
                    full_traj_aug = np.concatenate([obs_traj_aug, pred_traj_aug], axis=1)[:, ::-1]
                    obs_traj_aug = full_traj_aug[:, :obs_len]
                    pred_traj_aug = full_traj_aug[:, obs_len:]

                # Prompt generation
                scene_context = scene_desc[scene_id[0]]
                obs_traj_text_list = batch_traj2txt(obs_traj_aug)
                pred_traj_text_list = batch_traj2txt(pred_traj_aug)
                traj_context = " ".join([hist_ped_template.format(i, obs_traj_text_list[i], obs_len) for i in range(n_ped_temp)])
                if use_scene_context:
                    context = traj_context + " " + scene_context
                else:
                    context = traj_context

                if "forecast" in multimodal:
                    question = question_template["forecast"].format(0, pred_len)
                    obs_prompt = question_answer_template.format(question, context)
                    pred_prompt = answer_template["forecast"].format(0, pred_traj_text_list[0], pred_len)

                    processed_data["forecast"].append({"id": len(processed_data["forecast"]),
                                                       "type": "forecast",
                                                       "scene": scene_id[ped_id],
                                                       "frame": int(frame[ped_id]),
                                                       "n_ped": n_ped,
                                                       "ped_id": ped_id,
                                                       "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                                       "obs_traj": obs_traj_aug[0].tolist(),
                                                       "pred_traj": pred_traj_aug[0].tolist(),
                                                       "observation": obs_prompt, 
                                                       "forecast": pred_prompt})
            
            # Prompt generation
            scene_context = scene_desc[scene_id[0]]
            obs_traj_text_list = batch_traj2txt(obs_traj_trunc)
            traj_context = " ".join([hist_ped_template.format(i, obs_traj_text_list[i], obs_len) for i in range(n_ped_temp)])
            context = scene_context + " " + traj_context
                
            if "destination" in multimodal:
                question = question_template["destination"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                dest_coord = "(" + ", ".join(COORD_PRECISION[coord_system].format(pred_traj_trunc[0, -1, j]) for j in range(2)) + ")"
                pred_prompt = answer_template["destination"].format(0, dest_coord, pred_len)
                
                processed_data["destination"].append({"id": len(processed_data["destination"]), 
                                                      "type": "destination",
                                                      "scene": scene_id[ped_id],
                                                      "frame": int(frame[ped_id]),
                                                      "n_ped": n_ped,
                                                      "ped_id": ped_id,
                                                      "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                                      "obs_traj": obs_traj_trunc[0].tolist(),
                                                      "pred_traj": pred_traj_trunc[0].tolist(),
                                                      "observation": obs_prompt, 
                                                      "forecast": pred_prompt})
                
            if "direction" in multimodal:
                question = question_template["direction"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                direction = answer_template["direction_answer_list"][direction_type]
                pred_prompt = answer_template["direction"].format(0, direction)

                processed_data["direction"].append({"id": len(processed_data["direction"]), 
                                                "type": "direction",
                                                "scene": scene_id[ped_id],
                                                "frame": int(frame[ped_id]),
                                                "n_ped": n_ped,
                                                "ped_id": ped_id,
                                                "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                                "obs_traj": obs_traj_trunc[0].tolist(),
                                                "pred_traj": pred_traj_trunc[0].tolist(),
                                                "observation": obs_prompt, 
                                                "forecast": pred_prompt})
                
            if "group" in multimodal:
                question = question_template["group"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                if group_id is not None:
                    pred_prompt = answer_template["group"].format(0, group_id)
                else:
                    pred_prompt = answer_template["group_false"].format(0)

                processed_data["group"].append({"id": len(processed_data["group"]), 
                                                "type": "group",
                                                "scene": scene_id[ped_id],
                                                "frame": int(frame[ped_id]),
                                                "n_ped": n_ped,
                                                "ped_id": ped_id,
                                                "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                                "obs_traj": obs_traj_trunc[0].tolist(),
                                                "pred_traj": pred_traj_trunc[0].tolist(),
                                                "observation": obs_prompt, 
                                                "forecast": pred_prompt})
                
            if "collision" in multimodal:
                question = question_template["collision"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                if collision_id is not None:
                    pred_prompt = answer_template["collision"].format(0, collision_id)
                else:
                    pred_prompt = answer_template["collision_false"].format(0)

                processed_data["collision"].append({"id": len(processed_data["collision"]), 
                                                    "type": "collision",
                                                    "scene": scene_id[ped_id],
                                                    "frame": int(frame[ped_id]),
                                                    "n_ped": n_ped,
                                                    "ped_id": ped_id,
                                                    "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                                    "obs_traj": obs_traj_trunc[0].tolist(),
                                                    "pred_traj": pred_traj_trunc[0].tolist(),
                                                    "observation": obs_prompt, 
                                                    "forecast": pred_prompt})
                
            if "mimicry" in multimodal:
                question = question_template["mimicry"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                if mimicry_id is not None:
                    pred_prompt = answer_template["mimicry"].format(0, mimicry_id)
                else:
                    pred_prompt = answer_template["mimicry_false"].format(0)

                processed_data["mimicry"].append({"id": len(processed_data["mimicry"]), 
                                                  "type": "mimicry",
                                                  "scene": scene_id[ped_id],
                                                  "frame": int(frame[ped_id]),
                                                  "n_ped": n_ped,
                                                  "ped_id": ped_id,
                                                  "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                                  "obs_traj": obs_traj_trunc[0].tolist(),
                                                  "pred_traj": pred_traj_trunc[0].tolist(),
                                                  "observation": obs_prompt, 
                                                  "forecast": pred_prompt})

    processed_data_cat = []
    for m in multimodal:
        processed_data_cat += processed_data[m]

    with open(dst_file, encoding= "utf-8",mode="w") as file: 
        for i in tqdm(processed_data_cat, desc=f"Writing {dataset} {phase} dataset..."):
            file.write(json.dumps(round_floats(i)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="eth", type=str, help="dataset name")
    parser.add_argument('--phase', default="train", type=str, help="training phase")
    args = parser.parse_args()
    
    dataset = ["eth", "hotel", "univ", "zara1", "zara2"]
    phase = ["train", "val", "test"]
    assert args.dataset in dataset
    assert args.phase in phase
    dataset = [args.dataset]
    phase = [args.phase]
    
    obs_len = 8
    pred_len = 12

    for d in dataset:
        for p in phase:
            print(f"Processing {d} {p} dataset...")
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="meter", postfix="-multimodal")
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="meter", multimodal=["forecast"], postfix="")
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="meter", use_scene_context=False, postfix="-multimodal-nocontext")
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="meter", multimodal=["forecast"], use_scene_context=False, postfix="-nocontext")
            
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="pixel", postfix="-multimodal")
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="pixel", multimodal=["forecast"], postfix="")
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="pixel", use_scene_context=False, postfix="-multimodal-nocontext")
            preprocess_dataset(d, p, obs_len, pred_len, coord_system="pixel", multimodal=["forecast"], use_scene_context=False, postfix="-nocontext")
