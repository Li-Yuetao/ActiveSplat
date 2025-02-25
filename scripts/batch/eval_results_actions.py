#!/usr/bin/env python
import os
import sys
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
import argparse
import json

from tqdm import tqdm

from utils import PROJECT_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} results actions evaluator.')
    parser.add_argument('--results_dir',
                        type=str,
                        default=os.path.join(PACKAGE_PATH, 'results'),
                        help='Results directory.')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='Specify gpu id.')
    parser.add_argument('--force',
                        action='store_true',
                        help='Specify whether to force evaluation.')
    
    args = parser.parse_args()
    
    EVAL_PYTHON_SCRIPT_URL = os.path.join(PACKAGE_PATH, 'scripts', 'judges', 'eval_actions.py')
    
    user_config_url = os.path.join(PACKAGE_PATH, 'config', 'user_config.json')
    assert os.path.exists(user_config_url), f'User config not found: {user_config_url}'
    results_dir = args.results_dir
    force = args.force
    for result_dir in tqdm(os.listdir(results_dir)):
        result_dir_url = os.path.join(results_dir, result_dir)
        config_url = os.path.join(result_dir_url, 'config.json')
        with open(config_url, 'r') as f:
            config = json.load(f)
        if not os.path.exists(config['env']['config']):
            env_config = os.path.join(PACKAGE_PATH, 'config', 'env', os.path.basename(config['env']['config']))
            assert os.path.exists(env_config), f'Env config not found: {env_config}'
            config['env']['config'] = env_config
        with open(config_url, 'w') as f:
            json.dump(config, f, indent=4)
        actions_url = os.path.join(result_dir_url, 'actions.txt')
        if os.path.exists(config_url) and os.path.exists(actions_url):
            save_path = os.path.join(result_dir_url, 'actions_error.txt')
            if os.path.exists(save_path) and not force:
                print(f'{save_path} already exists, skip.')
                continue
            os.system(f'python {EVAL_PYTHON_SCRIPT_URL} --save_path {save_path} --config {config_url} --user_config {user_config_url} --gpu_id {args.gpu_id} --actions {actions_url} {"--force" if force else ""}')