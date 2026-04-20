# DiffusionLight-Video: Light estimation from videos using DiffusionLight with DepthAnything	

## Table of contents
-----
  * [Installation](#Installation)
  * [Getting-started](#Getting started)
  * [Prediction](#Prediction)
  * [Evaluation](#Evaluation)
  * [Citation](#Citation)
------

## Installation

To setup Conda on ssh, run the following commmands in the terminal:
```shell
ssh panther
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
```

To set up the Python environment you need to run the following commands:
```shell
conda env create -f environments/diffusionlight.yml
conda create -n depthanything python=3.10
conda activate depthanything
cd Depth-Anything-3/External/Depth-Anything-3
pip install xformers torch\>=2 torchvision
pip install -e .
cd ../../../
conda activate diffusionlight-video
pip install -r requirements.txt
```
Note that there are two Conda envrironments, one for DiffusionLight and one for DepthAnything.
Stay in the DiffusionLight environment and run any DepthAnything scripts in the DepthAnything environment using `conda run -n depthanything python script.py`, as is described in this README.

## Getting started

```shell
# Convert the video into individual frames:
python video_to_frames.py --video_file input/example.mov --output_dir input --framerate_reduction_factor 5
# Inpaint the chrome balls frame-by-frame
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/naive --video
# Generate frame-by-frame environment maps
python ball2envmap.py --ball_dir intermediate/ball_frames/naive/square --envmap_dir intermediate/ball_frames/naive/envmap
# Reconstruct chrome ball videos for visual analysis
python frames_to_video.py --input_dir intermediate/ball_frames/naive --output_dir intermediate/ball_videos/naive --fps 5
```

## Video methods

Make sure the video has no border or padding

```shell
python video_to_frames.py --video_file input/example.mov --output_dir input --framerate_reduction_factor 5
```
Naive:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/naive --video
python ball2envmap.py --ball_dir intermediate/ball_frames/naive/square --envmap_dir intermediate/ball_frames/naive/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/naive --output_dir intermediate/ball_videos/naive --fps 5
```
One seed:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/one-seed --video --one_seed
python ball2envmap.py --ball_dir intermediate/ball_frames/one-seed/square --envmap_dir intermediate/ball_frames/one-seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/one-seed --output_dir intermediate/ball_videos/one-seed --fps 5
```
One seed with custom seeds:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/one-seed --video --one_seed --seed "0,37,71"
python ball2envmap.py --ball_dir intermediate/ball_frames/one-seed/square --envmap_dir intermediate/ball_frames/one-seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/one-seed --output_dir intermediate/ball_videos/one-seed --fps 5 --seed "0,37,71"
```
Smooth:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/smooth --video --smooth_frames
python ball2envmap.py --ball_dir intermediate/ball_frames/smooth/square --envmap_dir intermediate/ball_frames/smooth/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/smooth --output_dir intermediate/ball_videos/smooth --fps 5
```
Smooth one seed:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/smooth_one_seed --video --one_seed --smooth_frames
python ball2envmap.py --ball_dir intermediate/ball_frames/smooth_one_seed/square --envmap_dir intermediate/ball_frames/smooth_one_seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/smooth_one_seed --output_dir intermediate/ball_videos/smooth_one_seed --fps 5
```
Smooth one seed with custom seeds:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/smooth_one_seed --video --one_seed --smooth_frames --seed "0,37,71"
python ball2envmap.py --ball_dir intermediate/ball_frames/smooth_one_seed/square --envmap_dir intermediate/ball_frames/smooth_one_seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/smooth_one_seed --output_dir intermediate/ball_videos/smooth_one_seed --fps 5 --seed "0,37,71"
```

Convert to HDR:
```shell
python exposure2hdr.py --input_dir <output_directory>/envmap --output_dir <output_directory>/hdr
```

## Citation

```
@inproceedings{Chinchuthakun2025DiffusionLightTurbo,
  author = {Chinchuthakun, Worameth and Phongthawee, Pakkapon and Raj, Amit and Jampani, Varun and Khungurn, Pramook and Suwajanakorn, Supasorn},
  title = {DiffusionLight-Turbo: Accelerated Light Probes for Free via Single-Pass Chrome Ball Inpainting},
  booktitle = {ArXiv},
  year = {2025},
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)


# Attribution
Example video from Vecteezy.com