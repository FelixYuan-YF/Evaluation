#!/bin/bash

# VBench 视频质量评估脚本
# 使用方法: 修改下面的变量后运行 bash evaluate.sh
export VBENCH_CACHE_DIR="/path/to/your/models" # 模型缓存目录

# ============== 配置区域 ==============
# --- 方式1: 传统模式 (指定视频文件夹) ---
VIDEOS_PATH="/path/to/your/videos"      # 视频文件夹路径

# --- 方式2: CSV输入模式 (指定CSV文件, 包含 video_path 列) ---
# CSV 格式示例:
#   video_path,prompt
#   /path/to/video1.mp4,a cat sitting on a chair
#   /path/to/video2.mp4,a dog running
INPUT_CSV=""  # CSV输入路径, 留空则使用传统模式

DIMENSION="subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality human_action temporal_style overall_consistency"
OUTPUT_PATH="./evaluation_results"
OUTPUT_CSV="./evaluation_results/results.csv"  # CSV输出路径, 留空则不输出CSV
MODE="custom_input"

# --- 多GPU配置 ---
NUM_GPUS=0  # GPU数量，设置为0则使用所有可用GPU
# ======================================

# 自动检测可用GPU数量
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "检测到 $NUM_GPUS 张GPU"
fi

echo "=========================================="
echo "VBench 视频质量评估"
echo "=========================================="

if [ -n "$INPUT_CSV" ]; then
    echo "输入CSV: $INPUT_CSV"
    echo "评估维度: $DIMENSION"
    echo "输出路径: $OUTPUT_PATH"
    echo "CSV输出: $OUTPUT_CSV"
    echo "GPU数量: $NUM_GPUS"

    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "使用多GPU分布式模式"
        CMD="torchrun --nproc_per_node=$NUM_GPUS eval_vbench.py \
            --input_csv \"$INPUT_CSV\" \
            --dimension $DIMENSION \
            --output_path \"$OUTPUT_PATH\" \
            --mode \"$MODE\" \
            --load_ckpt_from_local"
    else
        echo "使用单GPU模式"
        CMD="python eval_vbench.py \
            --input_csv \"$INPUT_CSV\" \
            --dimension $DIMENSION \
            --output_path \"$OUTPUT_PATH\" \
            --mode \"$MODE\" \
            --load_ckpt_from_local"
    fi

    if [ -n "$OUTPUT_CSV" ]; then
        CMD="$CMD --output_csv \"$OUTPUT_CSV\""
    fi
else
    echo "视频路径: $VIDEOS_PATH"
    echo "评估维度: $DIMENSION"
    echo "输出路径: $OUTPUT_PATH"
    echo "模式: $MODE"
    echo "GPU数量: $NUM_GPUS"

    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "使用多GPU分布式模式"
        CMD="torchrun --nproc_per_node=$NUM_GPUS evaluate.py \
            --videos_path \"$VIDEOS_PATH\" \
            --dimension $DIMENSION \
            --output_path \"$OUTPUT_PATH\" \
            --mode \"$MODE\" \
            --load_ckpt_from_local"
    else
        echo "使用单GPU模式"
        CMD="python evaluate.py \
            --videos_path \"$VIDEOS_PATH\" \
            --dimension $DIMENSION \
            --output_path \"$OUTPUT_PATH\" \
            --mode \"$MODE\" \
            --load_ckpt_from_local"
    fi

    if [ -n "$OUTPUT_CSV" ]; then
        CMD="$CMD --output_csv \"$OUTPUT_CSV\""
    fi
fi

echo "=========================================="
eval $CMD

echo "评估完成！结果保存在: $OUTPUT_PATH"