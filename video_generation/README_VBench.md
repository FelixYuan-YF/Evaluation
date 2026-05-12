# VBench - 视频生成质量评估工具

VBench 是一个用于评估视频生成模型综合能力的 benchmark 工具，支持评估任意视频的质量，可用于 T2V、I2V、V2V 等各种视频生成任务。

## 评估维度

VBench 评估包含以下 16 个维度：

### 质量维度 (Quality)
| 维度 | 说明 |
|------|------|
| subject_consistency | 主体一致性 |
| background_consistency | 背景一致性 |
| temporal_flickering | 时间闪烁 |
| motion_smoothness | 运动平滑度 |
| dynamic_degree | 动态程度 |
| aesthetic_quality | 美学质量 |
| imaging_quality | 成像质量 |

### 语义维度 (Semantic)
| 维度 | 说明 |
|------|------|
| object_class | 物体类别 |
| multiple_objects | 多物体 |
| human_action | 人类动作 |
| color | 色彩 |
| spatial_relationship | 空间关系 |
| scene | 场景 |
| temporal_style | 时间风格 |
| appearance_style | 外观风格 |
| overall_consistency | 整体一致性 |

## 安装

```bash
# 1. 安装 VBench
cd /path/to/VBench
pip install -e .

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 安装 detectron2（用于某些评估维度）
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git

# 4. (可选) 下载预训练模型到 ~/.cache/vbench/
# 预训练模型会在首次运行时自动下载，也可手动下载放至此目录
```

## 快速开始

### 评估自定义视频

```bash
# 方式一：评估单个维度
python evaluate.py \
    --dimension subject_consistency \
    --videos_path /path/to/你的视频文件夹 \
    --mode=custom_input \
    --output_path ./eval_results

# 方式二：评估多个维度（推荐）
python evaluate.py \
    --dimension subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    --videos_path /path/to/你的视频文件夹 \
    --mode=custom_input \
    --output_path ./eval_results
```

### 使用命令行工具

```bash
# 评估自定义视频
vbench evaluate \
    --dimension subject_consistency \
    --videos_path /path/to/视频 \
    --mode=custom_input

# 多 GPU 评估
vbench evaluate --ngpus=4 --dimension subject_consistency --videos_path /path/to/视频
```

### 使用 Python API

```python
from vbench import VBench

# 初始化
device = "cuda"  # 或 "cpu"
my_VBench = VBench(
    device,
    "vbench/VBench_full_info.json",
    "evaluation_results"
)

# 评估自定义视频
my_VBench.evaluate(
    videos_path="/path/to/你的视频文件夹",
    name="my_videos",
    dimension_list=[
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality"
    ],
    mode="custom_input"
)
```

## 视频文件组织

### 方式一：使用文件名作为 prompt
```
your_videos/
├── a_cat_running_in_the_park.mp4
├── a_dog_swimming_in_the_ocean.mp4
└── ...
```

### 方式二：使用 prompt 文件
```bash
python evaluate.py \
    --dimension subject_consistency \
    --videos_path /path/to/视频 \
    --prompt_file /path/to/prompts.txt \
    --mode=custom_input
```

prompt 文件格式（每行一个，文件名与 prompt 用空格分隔）：
```
video1.mp4 A cat running in the park
video2.mp4 A dog swimming in the ocean
```

## 计算总分

```bash
# 计算综合分数
python scripts/cal_final_score.py \
    --zip_file evaluation_results.zip \
    --model_name your_model_name
```
