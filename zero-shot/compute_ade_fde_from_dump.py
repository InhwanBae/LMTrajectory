import os
import json
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=1, type=int, help="dataset id")
parser.add_argument('--model', default=0, type=int, help="llm model id")
args = parser.parse_args()

# Data config
dataset = ['eth', 'hotel', 'univ', 'zara1', 'zara2'][args.dataset]
model = ['gpt-3.5-turbo-0301', 'gpt-4-0314', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'][args.model]
obs_len = 8
pred_len = 12
dump_file = './zero-shot/output_dump/{m}/{d}_chatgpt_api_dump.json'


if __name__ == '__main__':
    # Load the previous dump json file
    if os.path.exists(dump_file.format(m=model, d=dataset)):
        with open(dump_file.format(m=model, d=dataset), 'r') as f:
            output_dump = json.load(f)
    else:
        print("Dump file not found")
        exit()
    
    # Compute the ADE and FDE
    assert dataset == output_dump['dataset']
    print("Dataset:", output_dump['dataset'])
    print("Language model:", output_dump['llm_model'])
    print("Obs len / Pred len:", output_dump['obs_len'], '/', output_dump['pred_len'])
    print("Total number of scenes:", len(output_dump['data']))
    obs_len, pred_len = output_dump['obs_len'], output_dump['pred_len']

    ADE_list = []
    FDE_list = []
    for scene_id in tqdm(output_dump['data'].keys()):
        num_ped = output_dump['data'][scene_id]['num_ped']
        obs_traj = output_dump['data'][scene_id]['obs_traj']
        pred_traj = output_dump['data'][scene_id]['pred_traj']
        non_linear_ped = output_dump['data'][scene_id]['non_linear_ped']
        llm_processed = output_dump['data'][scene_id]['llm_processed']

        # Check the data
        assert num_ped == len(obs_traj)
        assert num_ped == len(pred_traj)
        assert num_ped == len(non_linear_ped)
        assert num_ped == len(llm_processed)
        assert all(len(llm_processed[i]) == 20 for i in range(num_ped))
        assert all(len(llm_processed[i][j]) == pred_len for i in range(num_ped) for j in range(20))
        assert all(len(llm_processed[i][j][k]) == 2 for i in range(num_ped) for j in range(20) for k in range(pred_len))
        
        obs_traj = np.array(obs_traj)
        pred_traj = np.array(pred_traj)
        non_linear_ped = np.array(non_linear_ped)
        llm_processed = np.array(llm_processed)

        # Filter the min ped num
        num_ped_threshold = 1
        if num_ped < num_ped_threshold:
            continue

        # Filter the non-linear ped
        non_linear = False
        if non_linear:
            num_ped = sum(non_linear_ped)
            obs_traj = obs_traj[non_linear_ped > 0.5]
            pred_traj = pred_traj[non_linear_ped > 0.5]
            llm_processed = llm_processed[non_linear_ped > 0.5]

        # Calculate ADE and FDE
        temp = np.linalg.norm(llm_processed - pred_traj[:, None, :, :], axis=-1)
        ADE = temp.mean(axis=-1).min(axis=-1)
        FDE = temp[..., -1].min(axis=-1)

        ADE_list.extend(ADE)
        FDE_list.extend(FDE)
    
    ADE_mean, ADE_std = np.mean(ADE_list), np.std(ADE_list)
    FDE_mean, FDE_std = np.mean(FDE_list), np.std(FDE_list)

    print("Dataset:", dataset, "Number of pedestrians:", len(ADE_list))
    print("MEAN ADE/FDE: {:.4f} / {:.4f}".format(ADE_mean, FDE_mean), end='   |   ')
    print("STD ADE/FDE: (+-{:.2f}) / (+-{:.2f})".format(ADE_std, FDE_std))
