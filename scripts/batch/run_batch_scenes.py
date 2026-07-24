#!/usr/bin/env python3
import os
import subprocess
import sys
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
SRC_PATH = os.path.abspath(os.path.join(WORKSPACE, 'src'))
sys.path.append(WORKSPACE)
sys.path.append(SRC_PATH)
from typing import Tuple

if __name__ == '__main__':
    datasets_config = [
        ('gibson.json', 'gibson_small.txt', 1000),
        ('gibson.json', 'gibson_big.txt', 2000),
        ('mp3d.json', 'mp3d_small.txt', 1000)]
    
    for config_file_name, scenes_file_name, step_num in datasets_config:
        config_file_url = os.path.join(WORKSPACE, 'config', 'datasets', config_file_name)
        with open(os.path.join(WORKSPACE, 'scripts', 'batch', scenes_file_name), 'r') as f:
            lines = f.readlines()
        for line in lines:
            scene_id = line.strip()
            result = subprocess.run([
                'ros2', 'launch', 'activesplat', 'habitat.launch.py',
                f'config:={config_file_url}', f'scene_id:={scene_id}',
                'hide_planner_windows:=1', 'hide_mapper_windows:=1',
                f'step_num:={step_num}'], check=False)
            if result.returncode != 0:
                raise SystemExit(
                    f'ActiveSplat failed for {scene_id} (exit {result.returncode})')