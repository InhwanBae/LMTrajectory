import json
import glob
from tqdm import tqdm
import argparse

from utils.compact_json_encoder import CompactJSONEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=1, type=int, help="dataset id")
parser.add_argument('--model', default=0, type=int, help="llm model id")
args = parser.parse_args()

# Data config
dataset = ['eth', 'hotel', 'univ', 'zara1', 'zara2'][args.dataset]
model = ['gpt-3.5-turbo-0301', 'gpt-4-0314', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'][args.model]
obs_len = 8
pred_len = 12
dump_filename_fragment = './zero-shot/output_dump/{0}/{0}*.json'.format(dataset)
dump_filename_output = './zero-shot/output_dump/{}/{}_chatgpt_api_dump.json'.format(model, dataset)


if __name__ == '__main__':
    # Gather the dump files
    dump_file_list = glob.glob(dump_filename_fragment)
    dump_file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    gathered_data = {'dataset': dataset,
                     'llm_model': None,
                     'prompt_template': None,
                     'obs_len': None,
                     'pred_len': None,
                     'data': {}}
    for dump_file in tqdm(dump_file_list):
        with open(dump_file, 'r') as f:
            dump_data = json.load(f)
        gathered_data['llm_model'] = dump_data['llm_model']
        gathered_data['prompt_template'] = dump_data['prompt_template']
        gathered_data['obs_len'] = dump_data['obs_len']
        gathered_data['pred_len'] = dump_data['pred_len']

        # Check data
        assert dataset == dump_data['dataset']
        assert obs_len == dump_data['obs_len']
        assert pred_len == dump_data['pred_len']

        for scene_id in dump_data['data'].keys():
            gathered_data['data'][scene_id] = dump_data['data'][scene_id]

            # Bug string finder
            num_ped = dump_data['data'][scene_id]['num_ped']
            obs_traj = dump_data['data'][scene_id]['obs_traj']
            pred_traj = dump_data['data'][scene_id]['pred_traj']
            non_linear_ped = dump_data['data'][scene_id]['non_linear_ped']
            llm_processed = dump_data['data'][scene_id]['llm_processed']
            
            is_there_bug = ''
            is_there_bug += '' if num_ped == len(obs_traj) else 'num_ped!=obs_traj, '
            is_there_bug += '' if num_ped == len(pred_traj) else 'num_ped!=pred_traj, '
            is_there_bug += '' if num_ped == len(non_linear_ped) else 'num_ped!=non_linear_ped, '
            is_there_bug += '' if num_ped == len(llm_processed) else 'num_ped!=llm_processed, '
            is_there_bug += '' if all(len(llm_processed[i]) == 20 for i in range(num_ped)) else 'sample!=20, '
            is_there_bug += '' if all(len(llm_processed[i][j]) == pred_len for i in range(num_ped) for j in range(len(llm_processed[i]))) else 'time!=pred_traj, '
            is_there_bug += '' if all(len(llm_processed[i][j][k]) == 2 for i in range(num_ped) for j in range(len(llm_processed[i])) for k in range(len(llm_processed[i][j]))) else 'coord!=2, '

            if is_there_bug != '':
                print('Bug found in dmup file {} scene_id {} log {}'.format(dump_file, scene_id, is_there_bug))

    # save the output
    with open(dump_filename_output, 'w') as f:
        json.dump(gathered_data, f, cls=CompactJSONEncoder, indent=2)
        # json.dump(gathered_data, f, indent=2)

    print('Fragmented dump files are merged.')
    