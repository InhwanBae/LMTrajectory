import os
import json
import time
import random
from tqdm import tqdm
import re
import argparse

import openai
from utils.compact_json_encoder import CompactJSONEncoder
from utils.trajectory_dataset import TrajectoryDataset

# Helpful links
# https://platform.openai.com/docs/guides/chat/introduction
# https://platform.openai.com/docs/guides/rate-limits/overview

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=1, type=int, help="dataset id")
parser.add_argument('--model', default=0, type=int, help="model id")
parser.add_argument('--scene_id', default=0, type=int, help="scene id for processing")
args = parser.parse_args()


# API config
OPENAI_API_KEY = ['sk-PASTE_YOUR_OPENAI_API_KEY_HERE000000000000000000'][0]
openai.api_key = OPENAI_API_KEY
model = ['gpt-3.5-turbo-0301', 'gpt-4-0314', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'][args.model]
temperature = 0.7
free_trial = False  # free tier = True, paid tier = False

# chatbot config
max_timeout = 20

# Data config
dataset = ['eth', 'hotel', 'univ', 'zara1', 'zara2'][args.dataset]
scene_idx = args.scene_id
obs_len = 8
pred_len = 12
dump_filename = './output_dump/{0}/{0}_chatgpt_api_dump_{1:04d}.json'.format(dataset, scene_idx)
                
prompt_system = "You are a helpful assistant that Extrapolate the coordinate sequence data."
# prompt_template_list = ["Forecast the next {1:d} (x, y) coordinates using the observed {0:d} (x, y) coordinate list.\nDirectly output the 5 different cases of the future {1:d} coordinate Python list without explanation.\nDo not repeat the same results.\n{2:s}",
prompt_template_list = ["Forecast the next {1:d} (x, y) coordinates using the observed {0:d} (x, y) coordinate list.\nDirectly output the 5 different cases of the future {1:d} coordinate Python list without explanation.\nList must be in single line, without line split or numbering.\nDirectly output final lists with 12 x-y coordinates each without for loop or multiply operators.\n{2:s}",
                        "Give me 5 more results other than the above methods.",
                        "Give me 5 more results other than the above methods.",
                        "Give me 5 more results other than the above methods."]
coord_template = "({0:.2f}, {1:.2f})"
scene_name_template = "scene_{:04d}"


if __name__ == '__main__':
    # Make dump file directory
    if not os.path.exists(os.path.dirname(dump_filename)):
        try:
            os.makedirs(os.path.dirname(dump_filename), exist_ok=True)
        except:
            if os.path.exists(os.path.dirname(dump_filename)):
                print('Directory already exists.')
            else:
                print('Error creating directory.')
                exit(1)

    # Load the previous dump json file
    if os.path.exists(dump_filename.format(dataset)):
        with open(dump_filename.format(dataset), 'r') as f:
            output_dump = json.load(f)
    else:
        output_dump = {'dataset': dataset, 
                       'llm_model': model, 
                       'prompt_template': prompt_template_list,
                       'obs_len': obs_len, 
                       'pred_len': pred_len, 
                       'data': {}}
    
    # Load the test dataset
    test_dataset = TrajectoryDataset('../datasets/{}/test/'.format(dataset), obs_len=obs_len, pred_len=pred_len, min_ped=0)
    num_scenes = len(test_dataset)
    scene_name = scene_name_template.format(scene_idx)

    if scene_idx >= num_scenes:
        print('Scene id is larger than the number of scenes in the test dataset.')
        exit(0)

    batch = test_dataset[scene_idx]
    obs_traj, pred_traj, non_linear_ped, _, _, _ = batch
    num_ped = obs_traj.shape[0]

    # Run the Chatbot model on the test dataset
    progressbar = tqdm(range(num_scenes))
    progressbar.set_description('Chatbot started!')
    progressbar.update(scene_idx)

    # Run the Chatbot model
    llm_response_list = [["" for _ in range(len(prompt_template_list))] for _ in range(num_ped)]
    llm_processed_list = [[] for _ in range(num_ped)]

    start_time = 0
    end_time = 0

    for ped_idx in range(num_ped):
        # Skip the ped_idx that already in the response
        if scene_name in output_dump['data'].keys():
            if len(output_dump['data'][scene_name]['llm_processed']) == num_ped:
                if len(output_dump['data'][scene_name]['llm_processed'][ped_idx]) == 20:
                    llm_response_list[ped_idx] = output_dump['data'][scene_name]['llm_output'][ped_idx]
                    llm_processed_list[ped_idx] = output_dump['data'][scene_name]['llm_processed'][ped_idx]
                    continue

        messages = [{"role": "system", "content": prompt_system}]

        for prompt_idx in range(len(prompt_template_list)):
            coord_str = '[' + ', '.join([coord_template.format(*obs_traj[ped_idx, i]) for i in range(obs_len)]) + ']'
            prompt = prompt_template_list[prompt_idx].format(obs_len, pred_len, coord_str)
            messages.append({"role": "user", "content": prompt})
            
            error_code = ''
            timeout = 0
            add_info = 0

            while timeout < max_timeout:
                # Prevent RateLimitError by waiting for 20 seconds
                end_time = time.time()
                if free_trial and start_time !=0 and end_time != 0 and end_time - start_time < 20:
                    time.sleep(20 - (end_time - start_time) + random.random() * 2)
                start_time = time.time()

                progressbar.set_description('Ped_id {}/{} Prompt_no. {}/{} retry {}/{} {}'.format(ped_idx+1, num_ped, prompt_idx+1, len(prompt_template_list), timeout, max_timeout, error_code))
                
                # Set additional information and settings when it kept failing
                tmp = 1.0 if timeout >= max_timeout // 2 else temperature
                if prompt_idx == 0 and timeout == max_timeout // 4 and add_info < 1:
                    messages[-1]['content'] += '\nProvide five hypothetical scenarios based on different extrapolation methods.'
                    add_info = 1
                elif prompt_idx == 0 and timeout == (max_timeout // 4) * 2 and add_info < 2:
                    messages[-1]['content'] += '\nYou can use methods like linear interpolation, polynomial fitting, moving average, and more.'
                    add_info = 2
                
                # Run the Chatbot model
                try:
                    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=tmp)
                    response = response['choices'][0]['message']['content']
                    # print('Coord_string--------------------\n', coord_str)
                    # print('Response------------------------\n', response)
                except Exception as err:
                    error_code = f"Unexpected {err=}, {type(err)=}"
                    print(error_code)
                    progressbar.set_description(f'{type(err)}, sleep for 30s')
                    time.sleep(random.random() * 30 + 30)
                    response = ''
                    timeout += 1
                    continue

                # filter out wrong answers
                if 'Rate limit reached' in response:
                    error_code = 'Rate limit reached'
                    progressbar.set_description('Rate limit reached, sleep for 20s')
                    time.sleep(random.random() * 20 + 20)
                    continue
                elif (not (abs(obs_traj[ped_idx, 0] - obs_traj[ped_idx, -1]).sum() < 0.3
                      or abs(obs_traj[ped_idx, 0] - obs_traj[ped_idx, 2]).sum() < 0.2
                      or abs(obs_traj[ped_idx, -3] - obs_traj[ped_idx, -1]).sum() < 0.2) and '[' + coord_template.format(*obs_traj[ped_idx, 0]) in response):
                    if prompt_idx == 0:
                        timeout += 1
                        error_code = 'Obs coordinates included'
                        continue
                elif ('[' + coord_template.format(*obs_traj[ped_idx, 0, ::-1]) in response
                      or '(x4, y4)]' in response
                      or ')]' not in response):
                    timeout += 1
                    error_code = 'Invalid response shape'
                    continue
                elif len(response) == 0:
                    timeout += 1
                    error_code = 'Empty response'
                    continue
                
                # Convert to list, check validity
                try:
                    response_cleanup = re.sub('[^0-9()\[\],.\-\n]', '', response.replace(':', '\n')).replace('(', '[').replace(')', ']')
                    response_cleanup = [eval(line) for line in response_cleanup.split('\n') if len(line) > 20 and line.startswith('[[') and line.endswith(']]')]
                except:
                    timeout += 1
                    error_code = 'Response to list failed'
                    continue
                
                # Remove repeated obs sequence or truncate the response
                if len(response_cleanup) >= 5:
                    response_cleanup = response_cleanup[-5:]
                
                # Check validity
                if (len(response_cleanup) == 5
                    and all(len(response_cleanup[i]) == pred_len for i in range(5))
                    and all(all(len(response_cleanup[i][j]) == 2 for j in range(pred_len)) for i in range(5))):
                    # Add the response to the dump list
                    llm_processed_list[ped_idx].extend(response_cleanup)
                    llm_response_list[ped_idx][prompt_idx] = response
                    messages.append({"role": "assistant", "content": response})
                    break
                else:
                    timeout += 1
                    error_code = 'Wrong response format'
                    continue

            if timeout >= max_timeout:
                print("Chatbot Timeout! Error scene_idx: {} ped_idx: {} prompt_idx: {}".format(scene_idx, ped_idx, prompt_idx))
                break

    # save the output
    output_scene = {'id': scene_idx, 
                    'num_ped': num_ped, 
                    'obs_traj': obs_traj.tolist(), 
                    'pred_traj': pred_traj.tolist(), 
                    'non_linear_ped': non_linear_ped.tolist(),
                    'llm_output': llm_response_list,
                    'llm_processed': llm_processed_list}
    output_dump['data'][scene_name] = output_scene

    with open(dump_filename.format(dataset), 'w') as f:
        json.dump(output_dump, f, cls=CompactJSONEncoder, indent=2)
    progressbar.set_description('Chatbot finished! Scene_idx: {}'.format(scene_idx))
    progressbar.update(1)
    progressbar.close()
    # print('Chatbot finished!')
