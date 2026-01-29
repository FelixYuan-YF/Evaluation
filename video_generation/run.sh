#!/bin/bash
CSV=[Replace with the path to the CSV file]
OUTPUT_DIR=[Directory for files generated during processing]
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=8

# 1. Extract frames
python extract_frames.py \
  --csv_path ${CSV} \
  --output_dir ${OUTPUT_DIR} \
  --num_workers $((GPU_NUM * 2)) \
  --num_frames 81 \
  --interval 5

# 2.1 Depth Estimation with Depth-Anything
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --standalone --nproc_per_node ${GPU_NUM} depth_estimation/Depth-Anything/inference_batch.py \
  --csv_path ${CSV} \
  --encoder vitl \
  --checkpoints_path checkpoints \
  --output_dir ${OUTPUT_DIR} \
  --bs 16 \
  --num_workers ${GPU_NUM}

# 2.2 Depth Estimation with UniDepth
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --standalone --nproc_per_node ${GPU_NUM} depth_estimation/UniDepth/inference_batch.py \
  --csv_path ${CSV} \
  --output_dir ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --bs 32 \
  --num_workers ${GPU_NUM}

# 3. Camera Tracking
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python camera_tracking/inference_batch.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))

# 4. eval pose
python eval_camera.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --output_csv ${OUTPUT_DIR}/eval_results.csv

# 5. eval CLIP-T and CLIP-F
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python eval_clip.py \
  --csv_path ${OUTPUT_DIR}/eval_results.csv \
  --gpu_num ${GPU_NUM} \
  --num_workers $((GPU_NUM * 2))

# 6. eval FVD

# 需要添加对于GT视频的抽帧，保存结构${OUTPUT_DIR_GT}/{id}/img/*.png

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python eval_fvd.py \
  --csv_path ${OUTPUT_DIR}/eval_results.csv \
  --gt_folder ${OUTPUT_DIR_GT} \
  --sample_folder ${OUTPUT_DIR}