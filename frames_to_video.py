import argparse
import cv2
import os
from tqdm import tqdm
import re

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True ,help='the folder containing the "raw" and "square" directories')
    parser.add_argument("--output_dir", default="", type=str, help="output directory")
    parser.add_argument("--fps", default=30, type=int, help="frames per second of output video")
    parser.add_argument("--seed", type=str, default="auto", help="put if the input frames end with '_seed[seed].png, as a comma-separated list'")
    return parser

def match(regex, st):
    return re.search(regex, st) is not None

def generate_video_from_frames(input_dir, output_dir, fps, seeds):
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    print(len(files))
    for seed in [int(x) for x in seeds.split(',')]:
        for ev in ["00", "25", "50"]:
            print(f"Seed {seed}, ev {ev}:")
            if seed == "auto":
                frame_regex = rf"_ev-{ev}\.png$"
            else:
                frame_regex = rf"_ev-{ev}_seed{str(seed)}\.png$"
            frames = [f for f in files if match(frame_regex, f)]
            frames.sort(key=lambda x: (len(x), x)) # Need the tuple so that example5 is sorted before example40
            suffix = "_ev-" + ev
            if seed != "auto":
                suffix += "_seed" + str(seed)
            video_path = output_dir + "/" + frames[0].split("0" + suffix + ".png")[0] + suffix + ".avi"
            frame = cv2.imread(os.path.join(input_dir, frames[0]))
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame_file in tqdm(frames):
                frame = cv2.imread(os.path.join(input_dir, frame_file))
                video.write(frame)
            video.release()
            cv2.destroyAllWindows()

def main():
    # load arguments
    args = create_argparser().parse_args()
    dir_contents = os.listdir(args.input_dir)

    for typ in ["raw", "square", "envmap"]:
        if typ not in dir_contents:
            print(f"input_dir doesn't contain '{typ}' folder, ignoring these")
        print("Generating video of type", typ)
        os.makedirs(os.path.join(args.output_dir, typ), exist_ok=True)
        generate_video_from_frames(
            args.input_dir + "/" + typ,
            args.output_dir + "/" + typ,
            args.fps,
            args.seed,
        )

if __name__ == "__main__":
    main()