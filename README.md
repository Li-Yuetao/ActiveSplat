<p align="center">

  <h2 align="center">ActiveSplat: High-Fidelity Scene Reconstruction<br>through Active Gaussian Splatting</h2>
  <p align="center">
    <a href="https://li-yuetao.github.io/"><strong>Yuetao Li</strong></a><sup>1,2*</sup>
    ·
    <a href="https://github.com/kzj18"><strong>Zijia Kuang</strong></a><sup>2*</sup>
    ·
    <a href="https://laura-ting.github.io/"><strong>Ting Li</strong></a><sup>2</sup>
    ·
    <a href="https://air.tsinghua.edu.cn/en/info/1046/1196.htm"><strong>Guyue Zhou</strong></a><sup>2</sup>
    ·
    <a href="https://scholar.google.nl/citations?hl=en&user=GDQ23eAAAAAJ&view_op=list_works"><strong>Shaohui Zhang</strong></a><sup>1</sup>
    ·
    <a href="https://zikeyan.github.io/"><strong>Zike Yan</strong></a><sup>2</sup>
  <p align="center">
        <sup>1</sup>Beijing Institute of Technology, <sup>2</sup>AIR, Tsinghua University
  </p>

<h3 align="center">
    <a href="https://arxiv.org/abs/2410.21955" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2410.21955-blue?logo=arxiv&color=%23B31B1B" alt="Paper arXiv"></a>
    <a href="https://li-yuetao.github.io/ActiveSplat/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Page-a" alt="Project Page"></a>
    <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</h3>
<div align="center"></div>

<div align=center> <img src="media/ui-x5.gif" width="850"/> </div>

<span class="dperact">ActiveSplat</span> enables the agent to explore the environment autonomously to build a 3D map on the fly. The integration of a Gaussian map and a Voronoi graph assures efficient and complete exploration with high-fidelity reconstruction results.

## 🛠️ Installation

Our environment has been tested on Ubuntu 20.04 with CUDA 11.8.

Clone the repository and create the conda environment:

```bash
mkdir -p ~/Workspace/activesplat_ws/src
git clone git@github.com:Li-Yuetao/ActiveSplat.git ~/Workspace/activesplat_ws/src/ActiveSplat && cd ~/Workspace/activesplat_ws/src/ActiveSplat
git submodule update --init --progress

conda env create -f environment.yaml
conda activate ActiveSplat
```

Install pytorch by following the [instructions](https://pytorch.org/get-started/locally/). For torch 2.0.1 with CUDA version 11.8:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Install diff-gaussian-rasterization (Note: You should be on CUDA version: 11.8)

```bash
cd ~/Workspace/activesplat_ws/src/ActiveSplat/submodules/diff-gaussian-rasterization
python setup.py install
pip install .
```

## 🖥️ Preparation

### Simulated environment

[Habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) need to be installed for simulation. We use v0.2.3 (`git checkout tags/v0.2.3`) for habitat-sim & habitat-lab and install the habitat-sim with the flag `--with-cuda`.

```bash
cd ~/Workspace/activesplat_ws/src/ActiveSplat/submodules/habitat/habitat-lab && git checkout tags/v0.2.3
pip install -e habitat-lab
pip install -e habitat-baselines
cd ~/Workspace/activesplat_ws/src/ActiveSplat/submodules/habitat/habitat-sim && git checkout tags/v0.2.3
# if you have bad network, you can use the following command to speed up
sed -i 's/https:\/\/github.com\//git@github.com:/g' .gitmodules # use `sed -i 's/git@github.com:/https:\/\/github.com\//g' .gitmodules` to restore
git submodule update --init --progress --recursive
python setup.py install --with-cuda
```

### Build

```bash
cd ~/Workspace/activesplat_ws/ && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
echo "source ~/Workspace/activesplat_ws/devel/setup.bash" >> ~/.bashrc
```

## 🚀 Run

### Config Datasets Path
Copy the `user_config.json` file from the `config/.templates` folder to the `config` folder, and set the paths for the [Gibson](https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform) and [MP3D](https://niessner.github.io/Matterport/#download) datasets in `user_config.json`.

<details>
  <summary>[Datasets folder structure (click to expand)]</summary>

```
  datasets_folder
    ├── gibson_habitat
    │   ├── gibson
    │   │   ├── Adrian.glb
    │   │   └── ...
    │   └── ...
    ├── matterport
    │   ├── v1
    │   │   ├── scans
    │   │   └── tasks
    │   |       ├── 1LXtFkjw3qL
    │   |       |   ├── 1LXtFkjw3qL.glb
    │   |       |   └── ...
    │   |       └── ...
    │   ├── v2
    |   └── ...
    └── ...
```
</details>

### Run ActiveSplat

#### Single scene
```bash
# If you want to save runtime data, you can add the `save_runtime_data:=1` flag
# e.g. Gibson-Denmark
roslaunch activesplat habitat.launch config:=config/datasets/gibson.json scene_id:=Denmark
# e.g. MP3D-pLe4wQe7qrG
roslaunch activesplat habitat.launch config:=config/datasets/mp3d.json scene_id:=pLe4wQe7qrG
```

#### Batch scenes
Here the system will run active mapping on 13 scenes ([gibson_small.txt](./scripts/batch/gibson_small.txt), [gibson_big.txt](./scripts/batch/gibson_big.txt), [mp3d_small.txt](./scripts/batch/mp3d_small.txt), [mp3d_big.txt](./scripts/batch/mp3d_big.txt)) in the Gibson & MP3D datasets. The results will be saved in the `results` folder.
```bash
bash scripts/batch/run_batch_scenes.sh
```

### Eval Results
Evaluate actions, this will read the actions from the `actions.txt` file in the result folder and evaluate them to generate the `actions_error.txt` file.
#### Single scene
```bash
result_name="2025-02-25_11-43-48_gibson_Denmark"
python scripts/judges/eval_actions.py --save_path results/$result_name/actions_error.txt --config results/$result_name/config.json --user_config config/user_config.json --actions results/$result_name/actions.txt --gpu_id 0
```

#### Batch scenes
```bash
# If you want to force re-evaluation, you can add the `--force` flag
python scripts/batch/eval_results_actions.py --results_dir ./results --gpu_id 0
```

## ✏️ Acknowledgments

Our implementation is built upon <a href="https://github.com/kzj18/activeINR-S">activeINR-S</a>. We would also like to thank the authors of the following open-source repositories:

- <a href="https://github.com/spla-tam/SplaTAM">SplaTAM</a> for the mapper implementation.
- <a href="https://github.com/muskie82/MonoGS">MonoGS</a> for the online gaussian map visualization.

If you find these works helpful, please consider citing them as well.

## 🎓 Citation

If you find our code/work useful in your research, please consider citing the following:
```bibtex
@article{li2024activesplat,
      title={ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting},
      author={Li, Yuetao and Kuang, Zijia and Li, Ting and Zhou, Guyue and Zhang, Shaohui and Yan, Zike},
      journal={arXiv preprint arXiv:2410.21955},
      year={2024}
}
```