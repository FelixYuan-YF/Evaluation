import glob
from argparse import ArgumentParser

import torch
import cv2
import pandas as pd
import numpy as np
from FVD.fvdcal import FVDCalculation
from torch import Tensor
from tqdm import tqdm


class MyFVDCalculation(FVDCalculation):
    def calculate_fvd_by_video_list(self, real_videos: Tensor, generated_videos: Tensor, model_path="FVD/model"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._load_model(model_path, device)

        fvd = self._compute_fvd_between_video(model, real_videos, generated_videos, device)

        return fvd.detach().cpu().numpy()


def load_videos(path):
    imgs = sorted(glob.glob(f"{path}/*.jpg"))
    frames = []
    for img in imgs:
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    # stack into numpy array first to preserve uint8 dtype, then convert to torch tensor
    frames = np.stack(frames, axis=0)  # (T, H, W, C), uint8
    frames = torch.from_numpy(frames).to(torch.uint8)
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames


def metric(gt_video_paths, sample_video_paths):
    gt_videos = torch.stack([load_videos(path) for path in tqdm(gt_video_paths, desc="loading real videos")])
    sample_videos = torch.stack([load_videos(path) for path in tqdm(sample_video_paths, desc="loading generated videos")])

    fvd_videogpt = MyFVDCalculation(method="videogpt")
    score_videogpt = fvd_videogpt.calculate_fvd_by_video_list(gt_videos, sample_videos)
    print(score_videogpt)

    fvd_stylegan = MyFVDCalculation(method="stylegan")
    score_stylegan = fvd_stylegan.calculate_fvd_by_video_list(gt_videos, sample_videos)
    print(score_stylegan)

    return score_videogpt, score_stylegan


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--gt_folder", type=str)
    parser.add_argument("--sample_folder", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    gt_video_paths = []
    sample_video_paths = []
    for index, row in df.iterrows():
        gt_video_paths.append(f"{args.gt_folder}/{row['id']}/img")
        sample_video_paths.append(f"{args.sample_folder}/{row['id']}/img")
    score_videogpt, score_stylegan = metric(gt_video_paths, sample_video_paths)
    print(f"FVD VideoGPT: {score_videogpt}, FVD StyleGAN: {score_stylegan}")    
    # write to txt
    with open("fvd_results.txt", "w") as f:
        f.write(f"FVD VideoGPT: {score_videogpt}\n")
        f.write(f"FVD StyleGAN: {score_stylegan}\n")