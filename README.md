# Functional-SLAM: Interaction-Aware Mapping with Online Functional Scene Graphs

<p align="center"><strong>Accepted at CoRL 2026</strong></p>

<p align="center">
  <strong>Xinggang Hu<sup>1,2</sup>, Chenyangguang Zhang<sup>3</sup>, Zihan Zhu<sup>3</sup>,
  Ruida Zhang<sup>1</sup>, Xiangkui Zhang<sup>2</sup>, Xiangyang Ji<sup>1,†</sup></strong><br>
  <sup>1</sup>Tsinghua University &nbsp;&nbsp;
  <sup>2</sup>Dalian University of Technology &nbsp;&nbsp;
  <sup>3</sup>ETH Zurich<br>
  <sup>†</sup>Corresponding author
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/xg-123/Functional-SLAM-dataset"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Dataset"></a>
  <a href="https://www.bilibili.com/video/BV13xbw6NECB/"><img src="https://img.shields.io/badge/Demo-Bilibili-00A1D6?logo=bilibili&logoColor=white" alt="Demo video"></a>
  <img src="https://img.shields.io/badge/Conference-CoRL%202026-6A5ACD" alt="CoRL 2026">
</p>

Functional-SLAM is an interaction-aware SLAM system that constructs an online
functional 3D scene graph while estimating camera motion and dense scene
geometry. It represents objects (O), functional carriers (C), robot-operable
interaction units (U), and their local and remote functional relations.

<p align="center">
  <a href="https://www.bilibili.com/video/BV13xbw6NECB/">
    <img src="https://i1.hdslb.com/bfs/archive/2f2a9309bfd831b29d39174a10f5bdcded01afa5.jpg" alt="Watch the Functional-SLAM demo" width="68%">
  </a><br>
  <a href="https://www.bilibili.com/video/BV13xbw6NECB/"><strong>Watch the demo video</strong></a>
</p>

## Motivation

Geometric and semantic SLAM systems provide increasingly rich maps, but they
usually do not model the small interaction elements and functional relations
needed for fine-grained robotic interaction. Recognizing a kettle, for example,
does not identify which handle should be grasped to lift or pour it.

Functional-SLAM maintains node geometry in anchor-keyframe coordinates and
stabilizes associations and relations using geometric, semantic, functional,
and multi-frame evidence. The resulting graph also provides functional-topology
candidates for loop closure, coupling functional mapping with pose estimation.

## Pipeline

<p align="center">
  <img src="Readme-Github/Functional-SLAM-Pipeline.png" alt="Functional-SLAM pipeline" width="100%">
</p>

The system consists of three coupled stages:

1. **Tracking and functional perception.** MASt3R-SLAM estimates poses,
   pointmaps, confidence, and keyframes. RAM++, language-model reasoning, and
   SAM3 produce frame-wise O/C/U observations and candidate relations.
2. **Online functional graph mapping.** Anchor-synchronized geometry, role-wise
   association, and temporal posterior updates maintain persistent nodes and
   commit stable local or remote functional edges.
3. **Functional-topology-assisted loop closure.** Functional graphlets
   supplement visual retrieval. All added candidates are geometrically verified
   before entering pose-graph optimization.

## Installation

The released code was developed on Ubuntu with Python 3.11, PyTorch 2.5.1,
CUDA 12.4, and an NVIDIA RTX 3090. A CUDA-capable GPU is required. Use a CUDA
toolkit compatible with the selected PyTorch build.

```bash
git clone --recursive git@github.com:Hbelief1998/Functional-SLAM-CoRL_2026.git
cd Functional-SLAM-CoRL_2026

conda create -n functional-slam python=3.11 -y
conda activate functional-slam

# Choose the CUDA build that matches your system. This is the CUDA 12.4 example.
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install -e recognize-anything
pip install -e sam3
pip install --no-build-isolation -e .
```

If the repository was cloned without `--recursive`, initialize Eigen before
building the CUDA backend:

```bash
git submodule update --init --recursive
```

## Checkpoints

Create one checkpoint directory and download the two MASt3R weights, the
retrieval codebook, RAM++, and SAM3:

```bash
mkdir -p checkpoints

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
  -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl \
  -P checkpoints/

hf download xinyu1205/recognize-anything-plus-model \
  ram_plus_swin_large_14m.pth --local-dir checkpoints

# SAM3 is gated. Request access at https://huggingface.co/facebook/sam3 and log in first.
hf auth login
hf download facebook/sam3 sam3.pt --local-dir checkpoints
```

Expected layout:

```text
checkpoints/
├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
├── ram_plus_swin_large_14m.pth
└── sam3.pt
```

## Dataset

The 18 FunGraph3D and 18 SceneFun3D RGB sequences used by this release are
available from the
[Functional-SLAM dataset](https://huggingface.co/datasets/xg-123/Functional-SLAM-dataset):

```bash
hf download xg-123/Functional-SLAM-dataset --repo-type dataset \
  --local-dir datasets/Functional-SLAM-dataset
```

Every selected sequence is exposed through an `rgb_SLAM` directory on the Hub:

```text
datasets/Functional-SLAM-dataset/
├── FunGraph3D/<scene>/<video>/rgb_SLAM/
└── SceneFun3D_Graph/<split>/<scene>/<sequence>/rgb_SLAM/
```

Sequence-specific camera calibration files are included in `config/`. Their
naming rules are:

```text
config/fungraph_<scene>_<video>_calib.yaml
config/scenefun3d_<split>_<scene>_<sequence>_rgb_SLAM_calib.yaml
```

## DeepSeek configuration

DeepSeek credentials are read only from environment variables; no API key is
stored in the repository. The paper experiments use the DeepSeek-V4-Flash
deployment. Set `DEEPSEEK_MODEL` to the corresponding deployment identifier
provided by your API endpoint.

```bash
export DEEPSEEK_API_KEY="your_api_key"
export DEEPSEEK_MODEL="your_deepseek_v4_flash_deployment"
```

You may instead pass `--deepseek-model` explicitly. The default API base URL is
`https://api.deepseek.com` and can be changed with `--deepseek-base-url`.

## Running Functional-SLAM

The configuration closest to the paper's full system is
`config/fungraph_eval_node_place.yaml`. It inherits the calibrated evaluation
settings and enables functional-topology-assisted place recognition.

Example on a FunGraph3D sequence:

```bash
python main.py \
  --dataset datasets/Functional-SLAM-dataset/FunGraph3D/0kitchen/video0/rgb_SLAM \
  --config config/fungraph_eval_node_place.yaml \
  --calib config/fungraph_0kitchen_video0_calib.yaml \
  --intrinsics-mode calib \
  --save-as fungraph_0kitchen_video0 \
  --no-viz \
  --semantic-out-dir logs/fungraph_0kitchen_video0/semantic \
  --enable-rampp \
  --enable-deepseek \
  --enable-sam3 \
  --sam3-u-parent-contain-thr 0.9
```

Example on a SceneFun3D sequence:

```bash
python main.py \
  --dataset datasets/Functional-SLAM-dataset/SceneFun3D_Graph/dev/420683/42445132/rgb_SLAM \
  --config config/fungraph_eval_node_place.yaml \
  --calib config/scenefun3d_dev_420683_42445132_rgb_SLAM_calib.yaml \
  --intrinsics-mode calib \
  --save-as scenefun3d_dev_420683_42445132 \
  --no-viz \
  --semantic-out-dir logs/scenefun3d_dev_420683_42445132/semantic \
  --enable-rampp \
  --enable-deepseek \
  --enable-sam3 \
  --sam3-u-parent-contain-thr 0.9
```

Remove `--no-viz` to open the interactive reconstruction window.

### Real-time configuration

The paper's real-time setting runs semantic and functional-graph updates only
on keyframes after scene lock and uses a smaller SAM3 processor resolution:

```bash
# Append these options to either command above.
--semantic-keyframes-only-after-scene-lock \
--functional-graph-keyframes-only-after-scene-lock \
--sam3-processor-resolution 672
```

For timing runs, per-frame visualization/debug artifacts can be disabled while
preserving the final map, trajectory, and functional graph:

```bash
--disable-intermediate-outputs \
--fps-profile-output logs/<run-name>/fps_profile.json
```

### Main outputs

For `--save-as <run-name>`, the principal outputs are:

```text
logs/<run-name>/<sequence>.txt                  # TUM-format keyframe trajectory
logs/<run-name>/<sequence>.ply                  # reconstructed point cloud
logs/<run-name>/semantic/online_functional_graph.json
logs/<run-name>/*_functional_graph_overlay.ply  # graph overlaid on the map
```

## Configuration overview

| File | Purpose |
|---|---|
| `config/base.yaml` | Base MASt3R-SLAM and matching parameters; functional place recognition is disabled. |
| `config/eval_calib.yaml` | Calibrated, single-thread evaluation settings. |
| `config/fungraph_eval.yaml` | Tracking and retrieval settings used for functional-scene evaluation. |
| `config/fungraph_eval_node_place.yaml` | Full functional-topology loop-retrieval configuration used for the paper system. |

## Implementation note

The released implementation follows the paper's core pipeline: MASt3R tracking,
open-vocabulary O/C/U perception, anchor-synchronized node geometry, role-wise
association, temporal relation stabilization, functional graphlets, and
geometrically verified functional loop candidates. Some thresholds and
candidate-selection gates are explicit engineering safeguards beyond the
compact equations in the paper. The API deployment name for the paper's
DeepSeek-V4-Flash model is intentionally configurable rather than hard-coded.

## Acknowledgements

This project builds on
[MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM),
[MASt3R](https://github.com/naver/mast3r),
[Recognize Anything](https://github.com/xinyu1205/recognize-anything), and
[SAM3](https://github.com/facebookresearch/sam3). Third-party components remain
subject to their respective licenses, included with their source code.

## License

Functional-SLAM is released under the CC BY-NC-SA 4.0 license. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{hu2026functional,
  title     = {Functional-SLAM: Interaction-Aware Mapping with Online Functional Scene Graphs},
  author    = {Hu, Xinggang and Zhang, Chenyangguang and Zhu, Zihan and Zhang, Ruida and Zhang, Xiangkui and Ji, Xiangyang},
  booktitle = {Conference on Robot Learning},
  year      = {2026}
}
```
