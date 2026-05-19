# DiffusionLight-Video: Spatially Varying Light Estimation from a Video

## Installation

To set up the Python environments you need to first set up Conda, then run the following commands (only need one of VGGT or DepthAnything unless you want to try both models):
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
conda create -n depthanything python=3.10.19
conda activate depthanything
cd Depth-Anything-3/External/Depth-Anything-3
pip install xformers torch==2.10.0 torchvision
pip install -e .
cd ../../../
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
Note that there are four different Conda envrironments, one for each part of the pipeline.

## Step 1 - DiffusionLight

Make sure the input video has no border or padding

Run the following commands:
```shell
conda activate diffusionlight-video
python scripts_depthlight/video_to_frames.py --video_file input/example.mov --output_dir input --framerate_reduction_factor 5
python inpaint.py --dataset input/example --output_dir intermediate/ball_frames --video
python ball2envmap.py --ball_dir intermediate/ball_frames/square --envmap_dir intermediate/ball_frames/envmap
python exposure2hdr.py --input_dir intermediate/ball_frames/envmap --output_dir intermediate/ball_frames/hdr
```
and optionally if you want to view the video output:
```shell
python scripts_depthlight/frames_to_video.py --input_dir intermediate/ball_frames --output_dir intermediate/ball_videos --fps 6
```
you can also try using the `smooth_frames` or `one_seed` flag on `inpaint.py` for the smoothing methods.

Alternatively, do all at once by running:
```shell
python run_diffusionlight.py --input input/example --ball_style naive
```
but you will not get proper logging updates.

## Step 2 - depth estimation

Either use DepthAnything or VGGT:
```shell
conda activate depthanything
python run_depth_anything.py --frames input/example --out_folder intermediate/depth --batch_size 32 --res 1024
```
or
```shell
conda activate vggt
python run_vggt.py --frames input/example --out_folder intermediate/depth --max_frames -1
```
If memory runs out, try decreasing the batch size, number of frames or image resolution.

Depth Anything 3 tends to perform poorly if the number of frames in the video is much larger than the batch size.

## Step 3 - run LEDiff
```shell
conda activate lediff
python LEDiff/examples/text_to_image/test_hdr_itm.py \
  --model_path LEDiff/model_highlight/ \
  --image_folder input/example/ \
  --output_hdr_path intermediate/LEDiff/ \
  --keep_size
```
you can also try using ``--model_path LEDiff/model_shadow/` for underexposed scenes

## Step 4 - process scene
```shell
conda activate diffusionlight-video
python process_scene.py \
  --ball_frames_folder intermediate/ball_frames \
  --depth_folder intermediate/depth \
  --lediff_folder intermediate/LEDiff \
  --out_folder output \
  --conf_quantile 0.1 \
  --voxel_size 0.005
```
Recommend voxel size of ~0.005 for VGGT or ~0.05 for DepthAnything.
Or run individual files in `scripts_depthlight` for more control:
- `depth_to_pointcloud.py`
- `make_lightcloud.npy`
- `make_backup_envmap.npy`
- `scale_lightcloud.npy`

## Step 5 - generate final environment maps at custom points
(still in diffusionlight-video env)
```shell
conda activate diffusionlight-video
python make_final_envmap.py \
  --lightcloud output/lightcloud.npy \
  --envmap output/missing_envmap.exr \
  --lightcloud_downscale output/lightcloud_downscale.npz \
  --out_folder final \
  --voxel_size_file intermediate/depth/voxel_size.npy \
  --downscale_type map
```

# Attribution
Example video from Vecteezy.com
