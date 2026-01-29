import torch
import torch.multiprocessing as mp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import json
import os
import math
import traceback
import time
from tqdm import tqdm

# --- 配置参数 ---
INPUT_DIRECTORY = "The Path to MotionBench folder(the one with subfolders self-collected and public-dataset)"
OUTPUT_DIRECTORY = "The Path to save results"
MODEL_NAME = "Model(e.g., Qwen/Qwen2.5-VL-3B-Instruct)"
MODEL_PATH = "The Path to the pretrained model checkpoint"
LORA_PATH = "The Path to the LoRA checkpoint (if any) or set to None"
VIDEO_PERSPECTIVE_FILE = "The Path to the video_info.meta.jsonl file"

NUM_GPUS = 8          # 使用多少块 GPU
PROCS_PER_GPU = 1     # 每块 GPU 启动多少个进程 (如果显存够大，可以设为 2-4)
# 总进程数 = NUM_GPUS * PROCS_PER_GPU

def analyze(input_file):
    """结果统计函数"""
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    all_dict = {"ALL": [0, 0]}
    with open(input_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                key = next(iter(item))
                for question in item[key]:
                    q_type = question.get("question_type", "unknown")
                    if q_type not in all_dict:
                        all_dict[q_type] = [0, 0]
                    all_dict[q_type][1] += 1
                    all_dict["ALL"][1] += 1
                    if question.get("judge") is True:
                        all_dict[q_type][0] += 1
                        all_dict["ALL"][0] += 1
            except Exception:
                continue

    print(f"\n--- Statistics for {input_file} ---")
    for key, value in all_dict.items():
        if value[1] > 0:
            score = round(value[0] / value[1] * 100, 2)
            print(f"{key}: {score}% ({value[0]}/{value[1]})")

def worker_proc(rank, world_size, data_shard):
    """工作进程：增加了详尽的错误处理和显存管理"""
    gpu_id = rank % NUM_GPUS
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Initializing on {device}...")
    
    model = None
    processor = None

    try:
        # 加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2"
        )
        if LORA_PATH:
            model = PeftModel.from_pretrained(model, LORA_PATH)
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ [Rank {rank}] Critical Error during model loading: {e}")
        traceback.print_exc()
        return

    output_filename = f"{OUTPUT_DIRECTORY}/{MODEL_NAME.split('/')[1]}_rank{rank}.jsonl"
    
    # 检查已处理视频（用于断点续传）
    processed_videos = set()
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                try:
                    processed_videos.update(json.loads(line).keys())
                except: continue

    with open(output_filename, 'a', encoding='utf-8') as output_file:
        for item in tqdm(data_shard, desc=f"Rank {rank}", position=rank):
            video_name = item.get("video_path", "")
            basename, _ = os.path.splitext(video_name)
            
            if not video_name or basename in processed_videos:
                continue
                
            video_path = os.path.join(INPUT_DIRECTORY, "self-collected", video_name)
            if not os.path.exists(video_path):
                video_path = os.path.join(INPUT_DIRECTORY, "public-dataset", video_name)
                if not os.path.exists(video_path):
                  print(f"[Rank {rank}] File not found: {video_path}")
                  continue

            try:
                question_type = item.get('question_type', 'default')
                value_list = []

                for question in item.get('qa', []):
                    correct_answer = question.get('answer', 'NA')
                    if correct_answer == "NA": 
                      continue

                    prompt = (f"Carefully watch the video and pay attention to temporal dynamics. "
                              f"Select the best option.\n{question['question']}\n"
                              f"Answer only the letter.")

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
                            {"type": "text", "text": prompt},
                        ],
                    }]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=128)
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    
                    # 简单判定逻辑
                    judge = True if correct_answer.upper() in output_text.upper()[:5] else False
                    value_list.append({
                        'question_type': question_type, 
                        'correct_answer': correct_answer, 
                        'output': output_text, 
                        'judge': judge
                    })
                    
                    # 及时手动清理 inputs 显存
                    del inputs, generated_ids
                    torch.cuda.empty_cache()

                if value_list:
                    json.dump({basename: value_list}, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    output_file.flush()

            except torch.cuda.OutOfMemoryError:
                print(f"⚠️ [Rank {rank}] OOM on {video_name}. Skipping...")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"❌ [Rank {rank}] Unexpected error on {video_name}: {e}")
                # traceback.print_exc() # 如果需要详细报错请取消注释
                continue

def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # 加载数据
    if not os.path.exists(VIDEO_PERSPECTIVE_FILE):
        print(f"Error: {VIDEO_PERSPECTIVE_FILE} not found.")
        return

    with open(VIDEO_PERSPECTIVE_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    world_size = NUM_GPUS * PROCS_PER_GPU
    avg = math.ceil(len(data) / world_size)
    shards = [data[i:i + avg] for i in range(0, len(data), avg)]

    print(f"Total items: {len(data)}, Total processes: {world_size}")
    
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        # 确保 shard 不为空
        p_data = shards[rank] if rank < len(shards) else []
        p = mp.Process(target=worker_proc, args=(rank, world_size, p_data))
        p.start()
        processes.append(p)

    # 主进程实时监控子进程状态，防止静默卡死
    while any(p.is_alive() for p in processes):
        for p in processes:
            if p.exitcode is not None and p.exitcode != 0:
                print(f"🚨 Process {p.pid} exited unexpectedly with code {p.exitcode}.")
        time.sleep(10)

    for p in processes:
        p.join()

    # 合并结果
    final_output = f"{OUTPUT_DIRECTORY}/{MODEL_NAME.split('/')[1]}.jsonl"
    with open(final_output, 'w', encoding='utf-8') as outfile:
        for rank in range(world_size):
            rank_file = f"{OUTPUT_DIRECTORY}/{MODEL_NAME.split('/')[1]}_rank{rank}.jsonl"
            if os.path.exists(rank_file):
                with open(rank_file, 'r') as f:
                    outfile.write(f.read())
                os.remove(rank_file) 

    print("\n✅ All processes finished.")
    analyze(final_output)

if __name__ == "__main__":
    main()