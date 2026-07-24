#!/usr/bin/env python3

import argparse
import json
import os
import sys
import threading
from typing import Union

import faulthandler
import numpy as np
import torch

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.logging import LoggingSeverity

from open3d.visualization import gui
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SCRIPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
for path in (os.path.join(SCRIPT_ROOT, "src"), os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.path_utils import source_root

PACKAGE_PATH = source_root(__file__)
SRC_PATH = os.path.join(PACKAGE_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mapper import MapperType
from utils import PROJECT_NAME, GlobalState
from dataloader.dataloader import get_dataset, HabitatDataset
from visualizer.visualizer import Visualizer


class MapperNode(Node):

    def __init__(self, args, device):
        super().__init__(f"{PROJECT_NAME}_mapper_node")

        self.get_logger().set_level(
            LoggingSeverity.DEBUG if args.debug else LoggingSeverity.INFO
        )

        self.declare_parameter("step_num", -1)

        self.args = args
        self.device = device

        self.visualizer = None
        self.gui_app = None

    def start_pipeline(self):

        args = self.args

        dataset = None

        os.chdir(PACKAGE_PATH)

        self.get_logger().info(
            f"Current working directory: {os.getcwd()}")

        with open(args.config) as f:
            config = json.load(f)

        step_num = int(self.get_parameter("step_num").value)
        if step_num >= 0:
            config["dataset"]["step_num"] = step_num

        if "env" in config:
            config["env"]["config"] = os.path.abspath(
                os.path.join(
                    os.path.dirname(args.config),
                    os.pardir,
                    os.pardir,
                    config["env"]["config"]))

        if "sensor" in config:
            config["sensor"]["config"] = os.path.abspath(
                os.path.join(
                    os.path.dirname(args.config),
                    os.pardir,
                    os.pardir,
                    config["sensor"]["config"]))

        with open(args.user_config) as f:
            user_config = json.load(f)

        dataset: Union[HabitatDataset] = get_dataset(
            config,
            user_config,
            args.scene_id,
            args.remark)

        hide_windows = bool(args.hide_windows)

        app = None
        if not hide_windows:
            app = gui.Application.instance
            app.initialize()

        self.visualizer = Visualizer(
            MapperType(args.mapper),
            args.config,
            GlobalState(args.mode),
            1 if hide_windows else app.add_font(
                gui.FontDescription(gui.FontDescription.MONOSPACE)),
            self.device,
            args.actions,
            dataset,
            bool(args.parallelized),
            hide_windows,
            bool(args.save_runtime_data),
            self)

        self.gui_app = app

if __name__ == '__main__':
    faulthandler.enable()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} mapper node.')
    parser.add_argument('--mapper',
                        type=str,
                        choices=list(MapperType.__members__),
                        required=True,
                        help='Specify the mapper type.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--scene_id',
                        type=str,
                        required=True,
                        help='Specify test scene id.')
    parser.add_argument('--user_config',
                        type=str,
                        required=True,
                        help='User config url (*.json).')
    parser.add_argument('--gpu_id',
                        type=int,
                        required=True,
                        help='Specify gpu id.')
    parser.add_argument('--mode',
                        type=str,
                        choices=list(GlobalState.__members__)[:-1],
                        required=True,
                        help='Specify the mode to start with.')
    parser.add_argument('--actions',
                        type=str,
                        required=True,
                        help='Specify the actions to replay.')
    parser.add_argument('--parallelized',
                        type=int,
                        required=True,
                        help='Tell the mapper node to be parallelized.')
    parser.add_argument('--hide_windows',
                        type=int,
                        required=True,
                        help='Disable windows.')
    parser.add_argument('--save_runtime_data',
                        type=int,
                        required=True,
                        help='Save runtime data.')
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='Debug mode, output more logs.')
    parser.add_argument('--remark',
                        type=str,
                        default='NONE',
                        help='remark info.')

    args, _ = parser.parse_known_args()

    if args.mode == "REPLAY" and args.actions is None:
        parser.error("Replay mode requires actions to replay.")

    device = torch.device(
        f"cuda:{args.gpu_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    rclpy.init()

    node = MapperNode(args, device)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True)

    spin_thread.start()

    try:

        node.start_pipeline()

        if node.gui_app is None:
            node.visualizer.wait_until_finished()
        else:
            node.gui_app.run()

    except KeyboardInterrupt:
        pass

    finally:
        try:
            executor.shutdown()
        except KeyboardInterrupt:
            pass
        spin_thread.join(timeout=5.0)
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass

    print(f"{PROJECT_NAME} mapper node finished.")