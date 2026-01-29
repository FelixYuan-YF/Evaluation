"""
Batch inference for camera tracking using multiple GPUs.

This module provides functionality for:
- Parallel camera tracking processing across multiple videos
- Multi-GPU support with automatic device assignment
- Subprocess management for camera tracking pipeline
- Progress tracking and error handling
"""

import pandas as pd
import os
import argparse
import concurrent.futures
from multiprocessing import Manager
import subprocess
import queue
from tqdm import tqdm
import clip
import torch
import cv2
from PIL import Image
import torch.multiprocessing as mp


def process_single_row(row, index, args, model=None, preprocess=None, device=None):
    video_path = row['infer_video_path']
    text = clip.tokenize([row['caption']]).to(device)
    text_features = model.encode_text(text)

    # 视频帧处理
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为PIL图像并预处理
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = preprocess(image).unsqueeze(0).to(device)
        # 提取帧特征
        with torch.no_grad():
            frame_feature = model.encode_image(image_input)
            frame_feature = frame_feature / frame_feature.norm(dim=-1, keepdim=True)
            frame_features.append(frame_feature)
    cap.release()
    frame_features = torch.cat(frame_features, dim=0)

    # 计算CLIP-T
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = (frame_features @ text_features.T).squeeze(1)
    clip_t = similarities.mean().item()

    # 计算CLIP-F
    pairwise_sims = torch.nn.functional.cosine_similarity(frame_features[:-1], frame_features[1:])
    clip_f = pairwise_sims.mean().item()

    return clip_t, clip_f



def worker(task_queue, result_queue,args, worker_id):
    device_id = worker_id % args.gpu_num
    device = f"cuda:{device_id}"
    model, preprocess = clip.load("ViT-B/32", device=device)
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        clip_t, clip_f = process_single_row(row, index, args, model, preprocess, device)
        result_queue.put((index, clip_t, clip_f))
        task_queue.task_done()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="Path to the csv file")
    parser.add_argument(
        "--gpu_num", type=int, default=8, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for parallel processing",
    )
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    df = pd.read_csv(args.csv_path)

    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # Add all tasks to queue
    for index, row in df.iterrows():
        task_queue.put((index, row))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        futures = []
        for id in range(args.num_workers):
            futures.append(executor.submit(worker, task_queue, result_queue, args, id))

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Finished workers",
        ):
            future.result()

    # Collect results
    while not result_queue.empty():
        index, clip_t, clip_f = result_queue.get()
        df.at[index, "clip_t"] = clip_t
        df.at[index, "clip_f"] = clip_f
    
    output_csv_path = os.path.splitext(args.csv_path)[0] + "_with_clip_scores.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
