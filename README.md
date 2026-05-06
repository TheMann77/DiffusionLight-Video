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

To set up the Python environments you need to run the following commands:
```shell
conda env create -f environments/diffusionlight.yml
conda activate diffusionlight-video
pip install -r requirements.txt
conda deactivate
conda create -n vggt python=3.10
conda activate vggt
cd vggt
pip install -r requirements.txt .
pip install natsort
cd ../
conda deactivate
conda create -n lediff python=3.10
conda activate lediff
cd LEDiff
pip install -e .
cd examples/text_to_image
pip install -r requirements.txt
pip install -r requirements_flax.txt
cd ../../../
conda deactivate
```
Then download the Highlight Hallucination Model model from [here](https://github.com/Hans1984/LEDiff/tree/main/examples/text_to_image) and unzip to `LEDiff`.
Note that there are three Conda envrironments, one for DiffusionLight, one for VGGT and one for LEDiff.

## Getting started

```shell
conda activate diffusionlight-video
# Convert the video into individual frames:
python video_to_frames.py --video_file input/example.mov --output_dir input --framerate_reduction_factor 5
# Inpaint the chrome balls frame-by-frame
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/naive --video
# Generate frame-by-frame environment maps
python ball2envmap.py --ball_dir intermediate/ball_frames/naive/square --envmap_dir intermediate/ball_frames/naive/envmap
# Reconstruct chrome ball videos for visual analysis
python frames_to_video.py --input_dir intermediate/ball_frames/naive --output_dir intermediate/ball_videos/naive --fps 6
# Compose HDR image
python exposure2hdr.py --input_dir intermediate/ball_frames/naive/envmap --output_dir intermediate/ball_frames/naive/hdr
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
python frames_to_video.py --input_dir intermediate/ball_frames/naive --output_dir intermediate/ball_videos/naive --fps 6
```
One seed:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/one-seed --video --one_seed
python ball2envmap.py --ball_dir intermediate/ball_frames/one-seed/square --envmap_dir intermediate/ball_frames/one-seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/one-seed --output_dir intermediate/ball_videos/one-seed --fps 6
```
One seed with custom seeds:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/one-seed --video --one_seed --seed "0,37,71"
python ball2envmap.py --ball_dir intermediate/ball_frames/one-seed/square --envmap_dir intermediate/ball_frames/one-seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/one-seed --output_dir intermediate/ball_videos/one-seed --fps 6 --seed "0,37,71"
```
Smooth:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/smooth --video --smooth_frames
python ball2envmap.py --ball_dir intermediate/ball_frames/smooth/square --envmap_dir intermediate/ball_frames/smooth/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/smooth --output_dir intermediate/ball_videos/smooth --fps 6
```
Smooth one seed:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/smooth_one_seed --video --one_seed --smooth_frames
python ball2envmap.py --ball_dir intermediate/ball_frames/smooth_one_seed/square --envmap_dir intermediate/ball_frames/smooth_one_seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/smooth_one_seed --output_dir intermediate/ball_videos/smooth_one_seed --fps 6
```
Smooth one seed with custom seeds:
```shell
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames/smooth_one_seed --video --one_seed --smooth_frames --seed "0,37,71"
python ball2envmap.py --ball_dir intermediate/ball_frames/smooth_one_seed/square --envmap_dir intermediate/ball_frames/smooth_one_seed/envmap
python frames_to_video.py --input_dir intermediate/ball_frames/smooth_one_seed --output_dir intermediate/ball_videos/smooth_one_seed --fps 6 --seed "0,37,71"
```

Convert to HDR:
```shell
python exposure2hdr.py --input_dir <output_directory>/envmap --output_dir <output_directory>/hdr
```

### LEDiff
```shell
python LEDiff/examples/text_to_image/test_hdr_itm.py \
  --model_path LEDiff/model_highlight/ \
  --image_folder input/example/ \
  --output_hdr_path intermediate/LEDiff/ \
  --keep_size
```

### DepthAnything3
Using the depthanything conda environment:

Generate depth-maps for raw frames:
Run `run_depth_anything.py` with `have_ball=False`

Generate depth-maps for frames with chrome balls:
Run `run_depth_anything.py` with `have_ball=True`, `ev=25`, `ball_type=naive` for example, or `one-seed`, `smooth`, etc.

Smooth chrome ball depth-maps with geometry:
Run `depth_anything_modifications.py` with `ev=25`

### Video DepthAnything
Generate depth-maps for raw frames:
```shell
cd Video-Depth-Anything
conda activate videodepthanything
python3 run.py --input_video ../input/example.mov --output_dir ../intermediate/depth_video/raw --encoder vitl --max_res=1024 --grayscale --save_npz
cd ../
conda activate diffusionlight-video
python video_to_frames.py --video_file intermediate/depth_video/raw/example_vis.mp4 --output_dir intermediate/depth_video/raw --output_filename depth --framerate_reduction_factor 5
```

### VGGT
Generate depthmaps, pointcloud and lightcloud:
Run `run_vggt.py`
Run `vggt_to_pointcloud.py`

### Generate output
Run `diffusionLight_to_lightcloud.py` with `ball_type="naive"`, or `"smooth"`, `"one-seed"`
Run `LEDiff_to_lightcloud.py`
Run `scale_lightcloud.py` with `ball_type` again
Run `make_final_envmap.py` with `ball_type` again, and set `relative_envmap_positions` for where in the scene you want to calculate.


# Attribution
Example video from Vecteezy.com