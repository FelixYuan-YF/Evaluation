import torch
import os
import csv
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from vbench import VBench
from vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json

VIDEO_EXTENSIONS = {'.mp4', '.gif'}

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/vbench/VBench_full_info.json',
        help="path to save the json file that contains the prompt and dimension information",
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        default=None,
        help="folder that contains the sampled videos (not needed when --input_csv is used)",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="CSV file with a 'video_path' column (and optional 'prompt' column) as input. "
             "When specified, --videos_path is derived from CSV automatically.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="path to save per-video evaluation results as CSV. "
             "Each row is a video, columns include video_path and one column per dimension.",
    )
    parser.add_argument(
        "--video_check_workers",
        type=int,
        default=None,
        help="number of worker threads for video readability checks. "
             "Default: min(32, os.cpu_count() + 4).",
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        action='store_true',
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )
    parser.add_argument(
        "--read_frame",
        type=bool,
        required=False,
        help="whether directly read frames, or directly read videos",
    )
    parser.add_argument(
        "--mode",
        choices=['custom_input', 'vbench_standard', 'vbench_category'],
        default='custom_input',
        help="""This flags determine the mode of evaluations, choose one of the following:
        1. "custom_input": receive input prompt from either --prompt/--prompt_file flags or the filename
        2. "vbench_standard": evaluate on standard prompt suite of VBench
        3. "vbench_category": evaluate on specific category
        """,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="None",
        help="""Specify the input prompt
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt_file.
        ** This option must be used with --mode=custom_input flag
        """
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        help="""Specify the path of the file that contains prompt lists
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt.
        ** This option must be used with --mode=custom_input flag
        """
    )
    parser.add_argument(
        "--category",
        type=str,
        required=False,
        help="""This is for mode=='vbench_category'
        The category to evaluate on, usage: --category=animal.
        """,
    )

    ## for dimension specific params ###
    parser.add_argument(
        "--imaging_quality_preprocessing_mode",
        type=str,
        required=False,
        default='longer',
        help="""This is for setting preprocessing in imaging_quality
        1. 'shorter': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
        2. 'longer': if the longer side is more than 512, the image is resized so that the longer side is 512.
        3. 'shorter_centercrop': if the shorter side is more than 512, the image is resized so that the shorter side is 512. 
        Then the center 512 x 512 after resized is used for evaluation.
        4. 'None': no preprocessing
        """,
    )
    args = parser.parse_args()
    return args


def read_input_csv(csv_path):
    """Read CSV file with video_path (and optional prompt) columns.

    Returns:
        videos_path: a directory containing all videos
                     (either the common parent, or a temp dir with symlinks)
        prompt_dict: {video_abs_path: prompt_str} if prompt column exists, otherwise {}
        video_paths: list of all video paths from CSV
        temp_dir: temporary directory to clean up later (or None if not created)
    """
    video_paths = []
    prompt_dict = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = row['video_path'].strip()
            video_paths.append(vp)
            if 'prompt' in row and row['prompt'].strip():
                prompt_dict[os.path.abspath(vp)] = row['prompt'].strip()

    if not video_paths:
        raise ValueError("CSV file contains no video_path entries")

    # In CSV mode, we pass the prompt_dict directly to VBench
    # VBench will use the absolute paths from prompt_dict keys, no need for videos_path
    videos_path = None  # Not needed for CSV mode with explicit paths
    
    return videos_path, prompt_dict, video_paths, None  # No temp_dir needed


def is_video_readable(video_path):
    """Return whether a video can be opened and at least one frame can be read."""
    if not os.path.isfile(video_path):
        return False, "file does not exist"

    if Path(video_path).suffix.lower() not in VIDEO_EXTENSIONS:
        return False, "unsupported video extension"

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return False, "cv2.VideoCapture failed to open the file"

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, _ = capture.read()
        if not success:
            return False, "failed to read the first frame"

        if frame_count == 0:
            return False, "video has zero frames"

        return True, ""
    except Exception as error:
        return False, str(error)
    finally:
        capture.release()


def get_video_check_worker_count(worker_count):
    """Resolve the number of worker threads for video readability checks."""
    if worker_count is not None:
        if worker_count < 1:
            raise ValueError('--video_check_workers must be greater than 0')
        return worker_count

    return min(32, (os.cpu_count() or 1) + 4)


def check_video_readability(video_path):
    """Check one video and return a normalized result tuple."""
    absolute_video_path = os.path.abspath(video_path)
    is_readable, reason = is_video_readable(absolute_video_path)
    return absolute_video_path, is_readable, reason


def filter_readable_video_paths(video_paths, worker_count=None):
    """Filter out unreadable videos with concurrent checks and an optional progress bar."""
    video_paths = list(video_paths)
    if not video_paths:
        raise ValueError('No candidate videos found for input validation')

    readable_video_paths = []
    skipped_video_count = 0
    max_workers = min(get_video_check_worker_count(worker_count), len(video_paths))
    print0(f'[Video check] Checking {len(video_paths)} videos with {max_workers} workers')

    check_results = [None] * len(video_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(check_video_readability, video_path): index
            for index, video_path in enumerate(video_paths)
        }
        completed_futures = as_completed(future_to_index)
        if tqdm is not None:
            completed_futures = tqdm(
                completed_futures,
                total=len(future_to_index),
                desc='Checking videos',
                unit='video',
            )

        for future in completed_futures:
            result_index = future_to_index[future]
            check_results[result_index] = future.result()

    for absolute_video_path, is_readable, reason in check_results:
        if is_readable:
            readable_video_paths.append(absolute_video_path)
            continue

        skipped_video_count += 1
        print0(f'[WARNING] Skip unreadable video: {absolute_video_path} ({reason})')

    print0(
        f'[Video check] {len(readable_video_paths)} readable videos, '
        f'{skipped_video_count} skipped videos'
    )

    if not readable_video_paths:
        raise ValueError('No readable videos found after input validation')

    return readable_video_paths


def create_filtered_video_directory(video_paths):
    """Create a temp directory containing symlinks to readable videos."""
    temp_dir = tempfile.mkdtemp(prefix='vbench_readable_videos_')
    used_filenames = set()

    for index, video_path in enumerate(video_paths):
        original_filename = os.path.basename(video_path)
        symlink_filename = original_filename
        if symlink_filename in used_filenames:
            stem = Path(original_filename).stem
            suffix = Path(original_filename).suffix
            symlink_filename = f'{stem}_{index}{suffix}'

        used_filenames.add(symlink_filename)
        symlink_path = os.path.join(temp_dir, symlink_filename)
        os.symlink(video_path, symlink_path)

    return temp_dir


def filter_csv_inputs(csv_prompt_dict, csv_video_paths, worker_count=None):
    """Filter CSV video paths and keep prompt mapping aligned with valid videos."""
    readable_video_paths = filter_readable_video_paths(csv_video_paths, worker_count)

    if not csv_prompt_dict:
        return readable_video_paths, {}

    filtered_prompt_dict = {}
    for video_path in readable_video_paths:
        if video_path in csv_prompt_dict:
            filtered_prompt_dict[video_path] = csv_prompt_dict[video_path]

    if not filtered_prompt_dict:
        raise ValueError('No readable videos with prompts found after input validation')

    return readable_video_paths, filtered_prompt_dict


def save_results_to_csv(results_dict, dimension_list, output_csv_path):
    """Save per-video evaluation results to a CSV file.

    Args:
        results_dict: {dimension: (avg_score, video_results_list)}
                      where video_results_list is [{'video_path': ..., 'video_results': score}, ...]
        dimension_list: list of evaluated dimensions
        output_csv_path: path to write the CSV
    """
    # Collect per-video scores: {video_path: {dimension: score}}
    video_scores = {}
    for dimension in dimension_list:
        if dimension not in results_dict:
            continue
        result = results_dict[dimension]
        # compute_* functions return (all_results, video_results)
        if isinstance(result, tuple) and len(result) == 2:
            _, video_results = result
        else:
            continue
        for item in video_results:
            vp = item['video_path']
            score = item['video_results']
            if vp not in video_scores:
                video_scores[vp] = {}
            video_scores[vp][dimension] = score

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['video_path'] + dimension_list
        writer.writerow(header)
        for vp in video_scores:
            row = [vp]
            for dim in dimension_list:
                row.append(video_scores[vp].get(dim, ''))
            writer.writerow(row)

    print0(f'CSV results saved to {output_csv_path}')


def main():
    args = parse_args()

    dist_init()
    print0(f'args: {args}')
    device = torch.device("cuda")
    my_VBench = VBench(device, args.full_json_dir, args.output_path)
    
    print0(f'start evaluation')

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    kwargs = {}

    prompt = []

    # --- CSV input mode ---
    temp_dir = None
    if args.input_csv is not None:
        if args.videos_path is not None:
            print0('[WARNING] --input_csv is specified, --videos_path will be ignored and derived from CSV')
        videos_path, csv_prompt_dict, csv_video_paths, temp_dir = read_input_csv(args.input_csv)
        readable_csv_video_paths, csv_prompt_dict = filter_csv_inputs(
            csv_prompt_dict,
            csv_video_paths,
            args.video_check_workers,
        )
        print0(f'[CSV mode] Loaded {len(csv_video_paths)} videos from {args.input_csv}')

        # If prompt column exists, use it
        if csv_prompt_dict:
            prompt = csv_prompt_dict
            args.mode = 'custom_input'
            print0(f'[CSV mode] Using prompts from CSV ({len(csv_prompt_dict)} entries)')
        else:
            videos_path = create_filtered_video_directory(readable_csv_video_paths)
            temp_dir = videos_path
            prompt = []
            args.mode = 'custom_input'
            print0(f'[CSV mode] Derived filtered videos_path: {videos_path}')
    else:
        if args.videos_path is None:
            raise ValueError("Either --videos_path or --input_csv must be specified")

        if os.path.isfile(args.videos_path):
            videos_path = filter_readable_video_paths(
                [args.videos_path],
                args.video_check_workers,
            )[0]
        else:
            candidate_video_paths = [
                os.path.join(args.videos_path, filename)
                for filename in os.listdir(args.videos_path)
                if Path(filename).suffix.lower() in VIDEO_EXTENSIONS
            ]
            readable_video_paths = filter_readable_video_paths(
                candidate_video_paths,
                args.video_check_workers,
            )
            videos_path = create_filtered_video_directory(readable_video_paths)
            temp_dir = videos_path
            print0(f'[Video check] Using filtered videos_path: {videos_path}')

    # --- Original prompt handling (non-CSV) ---
    if args.input_csv is None:
        if (args.prompt_file is not None) and (args.prompt != "None"):
            raise Exception("--prompt_file and --prompt cannot be used together")
        if (args.prompt_file is not None or args.prompt != "None") and (args.mode!='custom_input'):
            raise Exception("must set --mode=custom_input for using external prompt")

        if args.prompt_file:
            with open(args.prompt_file, 'r') as f:
                prompt = json.load(f)
            assert type(prompt) == dict, "Invalid prompt file format. The correct format is {\"video_path\": prompt, ... }"

            first_prompt_key = next(iter(prompt.keys())) if prompt else ""
            if os.path.isabs(first_prompt_key):
                _, prompt = filter_csv_inputs(
                    prompt,
                    list(prompt.keys()),
                    args.video_check_workers,
                )
                videos_path = None
        elif args.prompt != "None":
            prompt = [args.prompt]

    if args.category != "":
        kwargs['category'] = args.category

    kwargs['imaging_quality_preprocessing_mode'] = args.imaging_quality_preprocessing_mode

    results_dict = my_VBench.evaluate(
        videos_path = videos_path,
        name = f'results_{current_time}',
        prompt_list=prompt, # pass in [] to read prompt from filename
        dimension_list = args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.mode,
        **kwargs
    )

    # --- CSV output ---
    if args.output_csv is not None:
        save_results_to_csv(results_dict, args.dimension, args.output_csv)

    # Cleanup temp directory if created
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print0(f'[Video check] Cleaned up temp directory: {temp_dir}')

    print0('done')


if __name__ == "__main__":
    main()