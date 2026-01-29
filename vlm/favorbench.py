import torch
import torch.multiprocessing as mp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import json
import os
import math
from tqdm import tqdm

# 配置参数
INPUT_DIRECTORY = "The Path to FavorBench folder(the one with video files and video_perspective.json)"
OUTPUT_DIRECTORY = "The Path to save results"
MODEL_NAME = "Model(e.g., Qwen/Qwen2.5-VL-3B-Instruct)"
MODEL_PATH = "The Path to the pretrained model checkpoint"
LORA_PATH = "The Path to the LoRA checkpoint (if any) or set to None"
VIDEO_PERSPECTIVE_FILE = "The Path to the video_perspective.json file"

NUM_GPUS = 8          # 使用多少块 GPU
PROCS_PER_GPU = 2     # 每块 GPU 启动多少个进程 (如果显存够大，可以设为 2-4)
# 总进程数 = NUM_GPUS * PROCS_PER_GPU

def analyze(input_file):
    all_dict = {
        "ALL":[0,8184],
        "AS":[0,2637],
        "HAC":[0,1541],
        "SAD":[0,1662],
        "MAD":[0,1205],
        "CM":[0,1075],
        "NSM":[0,64]
    }
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            key = next(iter(item))
            for question in item[key]:
                task_type = question["task_type"]
                if question["judge"] == True:
                    all_dict[task_type][0] += 1
                    all_dict["ALL"][0] += 1

    scores1 = {key: round(value[0] / value[1] * 100, 2) for key, value in all_dict.items()}
    scores = [round(value[0] / value[1] * 100, 2) for value in all_dict.values()]
    formatted_output = " & ".join([f"{score}" for score in scores])
    print(input_file)
    for key, score in scores1.items():
        print(f"{key}: {score}%")
    print(formatted_output)

def worker_proc(rank, world_size, data_shard):
    """
    工作进程函数
    rank: 当前进程 ID
    world_size: 总进程数
    data_shard: 分配给当前进程的数据列表
    """
    # 计算当前进程应该使用的 GPU 编号
    gpu_id = rank % NUM_GPUS
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Loading model on {device}...")
    
    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device}, # 将模型强制加载到指定 GPU
        attn_implementation="flash_attention_2"
    )
    if LORA_PATH:
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    output_filename = f"{OUTPUT_DIRECTORY}/{MODEL_NAME.split('/')[1]}_rank{rank}.jsonl"
    
    # 检查断点续传
    processed_videos = set()
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                try:
                    processed_videos.update(json.loads(line).keys())
                except:
                    continue

    with open(output_filename, 'a', encoding='utf-8') as output_file:
        for item in tqdm(data_shard, desc=f"Rank {rank}", position=rank):
            video_name = item["video_name"]
            basename, _ = os.path.splitext(video_name)
            
            if basename in processed_videos:
                continue
                
            video_path = os.path.join(INPUT_DIRECTORY, video_name)
            if not os.path.exists(video_path):
                continue

            value_list = []
            for question in item['questions']:
                task_type = question['task_type']
                correct_answer = question['correct_answer']
                options = question['options']
                prompt = (f"Carefully watch the video and pay attention to temporal dynamics in this video, "
                          f"focusing on the camera motions, actions, activities, and interactions. "
                          f"Based on your observations, select the best option that accurately addresses the question.\n"
                          f"{question['question']}\nYou can only response with the answer among {question['options']}")

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
                        {"type": "text", "text": prompt},
                    ],
                }]
                
                # 数据处理
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                # 推理
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # 判断逻辑 (保持原样)
                containing_options = [opt for opt in options if opt != correct_answer and correct_answer in opt]
                if not containing_options:
                    judge = correct_answer.lower() in output_text.lower()
                else:
                    if correct_answer.lower() in output_text.lower():
                        judge = True
                        for option in containing_options:
                            if option.lower() in output_text.lower():
                                judge = False
                    else:
                        judge = False
                
                value_list.append({'task_type': task_type, 'correct_answer': correct_answer, 'output': output_text, 'judge': judge})

            # 写入结果
            new_item = {basename: value_list}
            json.dump(new_item, output_file, ensure_ascii=False)
            output_file.write('\n')
            output_file.flush() # 确保实时写入

def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # 加载总数据
    with open(VIDEO_PERSPECTIVE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 任务分片
    world_size = NUM_GPUS * PROCS_PER_GPU
    avg = math.ceil(len(data) / world_size)
    shards = [data[i:i + avg] for i in range(0, len(data), avg)]

    print(f"Total items: {len(data)}, Total processes: {world_size}")
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker_proc, args=(rank, world_size, shards[rank]))
        p.start()
        processes.append(p)

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
                os.remove(rank_file) # 合并后删除临时文件

    print("All processes finished. Merged result saved to:", final_output)
    analyze(final_output)

if __name__ == "__main__":
    main()
