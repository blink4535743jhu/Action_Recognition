import os
import cv2
import random
from glob import glob
from typing import List

# Constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # /content/Action_Recognition
INPUT_DIR = os.path.join(ROOT_DIR, "UCF50")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Preprocessed_UCF50")
FRAMES_PER_VIDEO = 16
RESIZE_SHAPE = (224, 224)

def get_frames(video_path: str, num_frames: int = 16, resize=(224, 224)) -> List:
    """Extracts uniformly sampled frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames == 0:
        print(f"⚠️  Warning: {video_path} has 0 frames.")
        return []

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

    while len(frames) < num_frames:
        frames.append(frames[-1])

    cap.release()
    return frames

def store_frames(frames: List, output_dir: str):
    """Saves the frames as JPEG images in the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"img_{i:03d}.jpg")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame_bgr)

def preprocess_dataset(input_dir: str, output_dir: str, num_frames: int = 16, resize=(224, 224)):
    """Preprocesses all videos in the dataset into folders of uniformly sampled frames."""
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(os.listdir(input_dir))

    for class_name in classes:
        class_input_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_input_path):
            continue

        video_files = glob(os.path.join(class_input_path, '*.avi'))
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_path = os.path.join(output_dir, class_name, video_name)

            if os.path.exists(os.path.join(video_output_path, "img_000.jpg")):
                print(f"⏭️  Skipping: {video_output_path} (already exists)")
                continue

            frames = get_frames(video_path, num_frames, resize)
            if frames:
                store_frames(frames, video_output_path)
                print(f"Saved: {video_output_path}")
            else:
                print(f"Skipped (no valid frames): {video_path}")

if __name__ == "__main__":
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR, FRAMES_PER_VIDEO, RESIZE_SHAPE)
