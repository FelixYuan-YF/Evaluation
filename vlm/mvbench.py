import os
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

# ==========================================
# 1. 配置
# ==========================================
model_path = "The Path to the pretrained model checkpoint"
mvbench_json_dir = "The Path to the MVBench json directory"
peft_model_path = "The Path to the LoRA checkpoint (if any) or set to None"
save_results_path = "./mvbench_results.json"
save_details_path = "./mvbench_details.jsonl" # 最终合并的详细结果

# MVBench 任务配置
DATA_LIST_CONFIG = {
    "Action Sequence": ("action_sequence.json", "MVBench/video/star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "MVBench/video/star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "MVBench/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "MVBench/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "MVBench/video/star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "MVBench/video/sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "MVBench/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "MVBench/video/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "MVBench/video/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "MVBench/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "MVBench/video/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "MVBench/video/tvqa/frames_fps3_hq/", "video", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "MVBench/video/clevrer/video_validation/", "video", False),
}

# ==========================================
# 2. 工具函数与类
# ==========================================
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_list = []
        for k, v in data_list.items():
            json_file = os.path.join(data_dir, v[0])
            if not os.path.exists(json_file): continue
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k, 'prefix': v[1], 'bound': v[3], 'data': data
                })

    def qa_template(self, data):
        question = f"Question: {data['question']}\nOptions:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer: answer_idx = idx
        return question, f"({chr(ord('A') + answer_idx)})"

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        video_path = os.path.abspath(os.path.join(item['prefix'], item['data']['video']))
        bound = (item['data']['start'], item['data']['end']) if item['bound'] else None
        question, gt_answer = self.qa_template(item['data'])
        return {
            'video_path': video_path, 'video_rel_path': item['data']['video'],
            'bound': bound, 'question': question, 'gt_answer': gt_answer,
            'task_type': item['task_type'], 'original_question': item['data']['question']
        }

def check_ans(pred, gt):
    if not pred or len(pred) == 0: return False
    pred_opt = pred.replace('(', '').replace(')', '').strip().upper()
    gt_opt = gt.replace('(', '').replace(')', '').strip().upper()
    return pred_opt[0] == gt_opt[0] if len(pred_opt) > 0 else False

# ==========================================
# 3. 工作进程函数
# ==========================================
def worker_task(rank, world_size, done_keys):
    """
    每个进程执行的函数
    """
    device = f"cuda:{rank}"
    print(f"进程 {rank} 正在加载模型至 {device}...")
    
    # 重新加载模型到特定显卡
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model = PeftModel.from_pretrained(model, peft_model_path)
    model = model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    
    dataset = MVBench_dataset(mvbench_json_dir, DATA_LIST_CONFIG)
    
    # 每个进程只跑一部分数据
    indices = [i for i in range(len(dataset)) if i % world_size == rank]
    
    tmp_save_path = f"{save_details_path}.tmp_{rank}"
    system_msg = "Carefully watch the video and select the best option."

    with open(tmp_save_path, 'w', encoding='utf-8') as f:
        for idx in tqdm(indices, desc=f"GPU {rank}"):
            sample = dataset[idx]
            unique_key = f"{sample['task_type']}_{sample['video_rel_path']}_{sample['original_question']}"
            
            # 如果主进程传来的已完成列表中有这个，直接跳过
            if unique_key in done_keys:
                continue

            try:
                # 推理逻辑
                video_path = sample["video_path"]
                user_q = sample["question"]
                if sample["bound"]:
                    user_q = f"Focus on {sample['bound'][0]}s-{sample['bound'][1]}s. {user_q}"

                messages = [{"role": "system", "content": [{"type": "text", "text": system_msg}]},
                            {"role": "user", "content": [{"type": "video", "video": video_path, "fps": 1.0},
                             {"type": "text", "text": user_q + "\nBest option:("}]}]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                                   padding=True, return_tensors="pt").to(device)

                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=5)
                    gen_text = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                
                pred_raw = "(" + gen_text.strip()
                is_correct = check_ans(pred_raw, sample['gt_answer'])

                res = {
                    "task_type": sample['task_type'], "video_rel_path": sample['video_rel_path'],
                    "original_question": sample['original_question'], "gt": sample['gt_answer'],
                    "pred_raw": pred_raw, "is_correct": is_correct
                }
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"GPU {rank} 处理出错: {e}")

# ==========================================
# 4. 主逻辑：合并结果与启动
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 张显卡，启动多进程评估...")

    # 1. 加载已有进度
    done_records = []
    done_keys = set()
    if os.path.exists(save_details_path):
        with open(save_details_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                done_records.append(record)
                done_keys.add(f"{record['task_type']}_{record['video_rel_path']}_{record['original_question']}")
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker_task, args=(rank, world_size, done_keys))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    # 3. 合并所有临时文件
    all_results = done_records
    for rank in range(world_size):
        tmp_path = f"{save_details_path}.tmp_{rank}"
        if os.path.exists(tmp_path):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    all_results.append(json.loads(line))
            os.remove(tmp_path) # 删除临时文件

    # 4. 写入最终详细文件
    with open(save_details_path, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 5. 统计分数
    acc_dict = {}
    for res in all_results:
        t = res['task_type']
        if t not in acc_dict: acc_dict[t] = [0, 0]
        acc_dict[t][1] += 1
        if res['is_correct']: acc_dict[t][0] += 1

    final_summary = {}
    print("\n" + "="*30 + "\n最终评测报告\n" + "="*30)
    total_c, total_s = 0, 0
    for task, (c, s) in acc_dict.items():
        acc = (c / s) * 100
        final_summary[task] = acc
        total_c += c; total_s += s
        print(f"{task:25s}: {acc:.2f}% ({c}/{s})")
    
    overall = (total_c / total_s) * 100 if total_s > 0 else 0
    final_summary["Overall_Avg"] = overall
    print(f"{'-'*30}\n总平均准确率: {overall:.2f}%")

    with open(save_results_path, 'w') as f:
        json.dump(final_summary, f, indent=4)
