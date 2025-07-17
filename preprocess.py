import os
import cv2
import random
from glob import glob
from typing import List
import argparse

def get_frames(video_path: str, num_frames: int = 16, resize=(224, 224)) -> List:
    """Extracts uniformly sampled frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames == 0:
        print(f"Warning: {video_path} has 0 frames.")
        return []

    # Uniform random sampling
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        indices = sorted(random.sample(range(total_frames), num_frames))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)

    # Pad if not enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    cap.release()
    return frames

def store_frames(frames: List, output_dir: str):
    """Stores a list of frames as sequential JPEG files in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"img_{i:03d}.jpg")
        # Convert back to BGR for saving with OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame_bgr)

def preprocess_dataset(input_dir: str, output_dir: str, num_frames: int = 16, resize=(224, 224)):
    """Processes all videos in the UCF50 dataset into frame folders."""
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(os.listdir(input_dir))

    for class_name in classes:
        class_input_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_input_path):
            continue

        video_files = glob(os.path.join(class_input_path, '*.avi'))

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            relative_output_path = os.path.join(output_dir, class_name, video_name)

            if os.path.exists(os.path.join(relative_output_path, f"img_000.jpg")):
                print(f"Skipping already processed: {relative_output_path}")
                continue

            frames = get_frames(video_path, num_frames=num_frames, resize=resize)
            if frames:
                store_frames(frames, relative_output_path)
                print(f"✅ Processed: {video_path} → {relative_output_path}")
            else:
                print(f"⚠️  Skipped (no frames): {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UCF50 videos into 16-frame image folders.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw UCF50 video dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to store extracted frame folders.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to extract per video.")
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.output_dir, num_frames=args.num_frames)
