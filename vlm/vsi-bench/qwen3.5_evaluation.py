import os
import json
import jsonlines
import argparse
import gc
import multiprocessing as mp
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from threading import Lock

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from vsi_util import *


# ============ 多进程预处理 ============

_worker_processor: Optional[AutoProcessor] = None


def _worker_init(model_path: str):
    """
    每个子进程启动时执行一次。
    在子进程中加载 AutoProcessor，避免跨进程序列化。
    """
    global _worker_processor
    _worker_processor = AutoProcessor.from_pretrained(model_path)


def _worker_preprocess(args: Tuple[str, str, float, bool]) -> Dict[str, Any]:
    """
    在子进程中运行的预处理函数。
    返回可 pickle 的 dict（不含不可序列化对象）。
    """
    video_path, prompt, fps, enable_thinking = args
    global _worker_processor

    try:
        # ---------- 视频解码 ----------
        messages_for_vision = [{
            "role": "user",
            "content": [{
                "type": "video",
                "video": video_path,
                "fps": fps,
                "max_pixels": 360 * 640,
            }]
        }]

        _image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages_for_vision,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        del _image_inputs, messages_for_vision

        if not video_inputs:
            raise ValueError("process_vision_info returned empty video_inputs")

        # ---------- 构造 prompt ----------
        final_messages = [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt}
            ]}
        ]

        # enable_thinking: Qwen3.5 特有参数
        text_prompt = _worker_processor.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        del final_messages

        # ---------- 组装结果（必须全部可 pickle） ----------
        llm_input = {
            "prompt": text_prompt,
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs
        }
        del text_prompt, video_inputs, video_kwargs

        return {
            "video_path": video_path,
            "llm_input": llm_input,
            "prompt": prompt,
            "error": None
        }

    except Exception as e:
        return {
            "video_path": video_path,
            "llm_input": None,
            "prompt": prompt,
            "error": str(e)
        }


class PreprocessResult:
    """轻量包装，方便主进程使用。"""
    __slots__ = ['video_path', 'llm_input', 'prompt', 'error']

    def __init__(self, video_path: str, prompt: str,
                 llm_input: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None):
        self.video_path = video_path
        self.prompt = prompt
        self.llm_input = llm_input
        self.error = error

    def release(self):
        self.llm_input = None
        self.error = None


class AsyncPreprocessPipeline:
    """
    基于 ProcessPoolExecutor 的多进程预处理流水线。
    """

    def __init__(self, model_path: str, fps: float = 2.0,
                 enable_thinking: bool = False,
                 prefetch_count: int = 8, num_workers: int = 4):
        self.fps = fps
        self.enable_thinking = enable_thinking
        self.prefetch_count = prefetch_count
        self.num_workers = num_workers

        # 使用 'spawn' 上下文，避免 fork 后 CUDA 状态异常
        ctx = mp.get_context('spawn')

        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(model_path,),
        )
        self._pending: deque[Tuple[str, Future]] = deque()
        self._shutdown = False

    def submit(self, video_path: str, prompt: str):
        if self._shutdown:
            raise RuntimeError("Pipeline already shut down")
        future = self._executor.submit(
            _worker_preprocess,
            (video_path, prompt, self.fps, self.enable_thinking)
        )
        self._pending.append((video_path, future))

    def get_next(self) -> Optional[PreprocessResult]:
        if not self._pending:
            return None
        video_path, future = self._pending.popleft()
        try:
            raw: Dict[str, Any] = future.result(timeout=600)
            return PreprocessResult(
                video_path=raw["video_path"],
                prompt=raw["prompt"],
                llm_input=raw.get("llm_input"),
                error=raw.get("error")
            )
        except Exception as e:
            return PreprocessResult(video_path=video_path, prompt="", error=str(e))

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def shutdown(self):
        self._shutdown = True
        for _, future in self._pending:
            future.cancel()
        self._pending.clear()
        self._executor.shutdown(wait=True)


# ============ 批量处理 ============

def process_vsibench_batch(
    vsibench: pd.DataFrame,
    model: LLM,
    model_path: str,
    res_path: str,
    video_path_root: str,
    sampling_params: SamplingParams,
    fps: float = 2.0,
    enable_thinking: bool = False,
    prefetch_count: int = 8,
    num_workers: int = 4,
    batch_size: int = 4
):
    """
    流水线化的批量处理 VSI-Bench 数据集。
    """
    # 断点续传
    prev_id = []
    if os.path.exists(os.path.join(res_path, 'response.jsonl')):
        with jsonlines.open(os.path.join(res_path, 'response.jsonl')) as reader:
            prev_response = list(reader)
            prev_id = [item['id'] for item in prev_response]
        print(f"已处理 {len(prev_id)} 个样本，跳过")

    # 筛选未处理的数据
    remaining_indices = []
    remaining_data = []
    for i in range(len(vsibench)):
        cur_data = vsibench.loc[i]
        if int(cur_data['id']) not in prev_id:
            remaining_indices.append(i)
            remaining_data.append(cur_data)

    print(f"待处理: {len(remaining_data)} 个样本 (batch_size={batch_size})")

    if not remaining_data:
        print("没有需要处理的样本")
        return

    csv_lock = Lock()
    save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="saver")
    save_futures: deque[Future] = deque()

    def _save_result(idx: int, data: dict, response: str, res_path: str):
        with csv_lock:
            with open(os.path.join(res_path, 'response.jsonl'), "a") as f:
                f.write(
                    json.dumps({
                        "id": int(data["id"]),
                        "predicted_answer": response,
                        'dataset': data['dataset'],
                        'scene_name': data['scene_name'],
                        'question': data['question'],
                        'ground_truth': data['ground_truth'],
                        'question_type': data['question_type'],
                        'prompt': data.get('prompt', ''),
                    }, ensure_ascii=False) + "\n"
                )

    def _drain_save_futures():
        while save_futures:
            if save_futures[0].done():
                fut = save_futures.popleft()
                try:
                    fut.result()
                except Exception as e:
                    print(f"⚠️ 保存时出错: {e}")
            else:
                break

    # 主流水线
    pipeline = AsyncPreprocessPipeline(
        model_path=model_path,
        fps=fps,
        enable_thinking=enable_thinking,
        prefetch_count=prefetch_count,
        num_workers=num_workers
    )

    try:
        # 预填充
        initial_submit = min(prefetch_count + batch_size, len(remaining_data))
        for i in range(initial_submit):
            cur_data = remaining_data[i]
            prompt = "These are frames of a video.\n" + cur_data['question']
            if cur_data['options'] is None:
                prompt += "\nPlease answer the question using a numerical value (e.g., 42 or 3.1)."
            else:
                options = cur_data['options'].tolist()
                prompt += "\nOptions:\n" + "\n".join(options)
                prompt += "\nAnswer with the option's letter from the given choices directly."

            video_url = os.path.join(video_path_root, f"{cur_data['dataset']}", f"{cur_data['scene_name']}.mp4")
            pipeline.submit(video_url, prompt)

        next_to_submit = initial_submit
        total_processed = 0

        pbar = tqdm(total=len(remaining_data), desc="VSI-Bench")

        while total_processed < len(remaining_data):
            # Phase 1: 收集一批预处理结果
            batch_results: List[PreprocessResult] = []
            batch_inputs: List[Dict[str, Any]] = []
            batch_indices: List[int] = []
            batch_data: List[dict] = []
            failed_in_preprocess: List[Tuple[int, dict, str]] = []

            collect_target = min(batch_size, len(remaining_data) - total_processed)

            while len(batch_results) + len(failed_in_preprocess) < collect_target:
                if pipeline.pending_count == 0:
                    if next_to_submit < len(remaining_data):
                        cur_data = remaining_data[next_to_submit]
                        prompt = "These are frames of a video.\n" + cur_data['question']
                        if cur_data['options'] is None:
                            prompt += "\nPlease answer the question using a numerical value (e.g., 42 or 3.1)."
                        else:
                            options = cur_data['options'].tolist()
                            prompt += "\nOptions:\n" + "\n".join(options)
                            prompt += "\nAnswer with the option's letter from the given choices directly."

                        video_url = os.path.join(video_path_root, f"{cur_data['dataset']}", f"{cur_data['scene_name']}.mp4")
                        pipeline.submit(video_url, prompt)
                        next_to_submit += 1
                    else:
                        break

                prep_result = pipeline.get_next()
                if prep_result is None:
                    if next_to_submit >= len(remaining_data) and pipeline.pending_count == 0:
                        break
                    continue

                if prep_result.error is not None:
                    print(f"❌ Preprocess error {prep_result.video_path}: {prep_result.error}")
                    idx = remaining_indices[total_processed + len(batch_results) + len(failed_in_preprocess)]
                    data = remaining_data[total_processed + len(batch_results) + len(failed_in_preprocess)]
                    failed_in_preprocess.append((idx, data, prep_result.error))
                    prep_result.release()
                else:
                    batch_results.append(prep_result)
                    batch_inputs.append(prep_result.llm_input)
                    idx = remaining_indices[total_processed + len(batch_results) - 1]
                    batch_indices.append(idx)
                    batch_data.append(remaining_data[total_processed + len(batch_results) - 1])

                if next_to_submit < len(remaining_data):
                    cur_data = remaining_data[next_to_submit]
                    prompt = "These are frames of a video.\n" + cur_data['question']
                    if cur_data['options'] is None:
                        prompt += "\nPlease answer the question using a numerical value (e.g., 42 or 3.1)."
                    else:
                        options = cur_data['options'].tolist()
                        prompt += "\nOptions:\n" + "\n".join(options)
                        prompt += "\nAnswer with the option's letter from the given choices directly."

                    video_url = os.path.join(video_path_root, f"{cur_data['dataset']}", f"{cur_data['scene_name']}.mp4")
                    pipeline.submit(video_url, prompt)
                    next_to_submit += 1

            # 更新失败计数
            num_failed = len(failed_in_preprocess)
            if num_failed > 0:
                # 保存失败记录
                for idx, data, error in failed_in_preprocess:
                    save_futures.append(
                        save_executor.submit(_save_result, idx, data, f"ERROR: {error}", res_path)
                    )
                total_processed += num_failed
                pbar.update(num_failed)

            if not batch_inputs:
                if pipeline.pending_count == 0 and next_to_submit >= len(remaining_data):
                    break
                continue

            # Phase 2: 批量 GPU 推理
            current_batch_size = len(batch_inputs)
            try:
                outputs = model.generate(batch_inputs, sampling_params=sampling_params)

                for prep in batch_results:
                    prep.release()
                del batch_inputs

                # Phase 3: 异步保存结果
                for i, output in enumerate(outputs):
                    generated_text = output.outputs[0].text.strip()

                    # 处理选项型问题
                    cur_data_dict = batch_data[i].to_dict() if hasattr(batch_data[i], 'to_dict') else dict(batch_data[i])
                    if cur_data_dict.get('options') is not None:
                        options_list = cur_data_dict['options']
                        if hasattr(options_list, 'tolist'):
                            options_list = options_list.tolist()
                        if generated_text in options_list:
                            generated_text = generated_text.split('.')[0]

                    save_futures.append(
                        save_executor.submit(
                            _save_result, batch_indices[i], cur_data_dict, generated_text, res_path
                        )
                    )

                del outputs
                total_processed += current_batch_size
                pbar.update(current_batch_size)

            except Exception as e:
                print(f"❌ Batch inference error: {e}")
                import traceback
                traceback.print_exc()

                for i, prep in enumerate(batch_results):
                    save_futures.append(
                        save_executor.submit(
                            _save_result, batch_indices[i],
                            batch_data[i].to_dict() if hasattr(batch_data[i], 'to_dict') else dict(batch_data[i]),
                            f"ERROR: {str(e)}", res_path
                        )
                    )
                    prep.release()
                del batch_inputs

                total_processed += current_batch_size
                pbar.update(current_batch_size)

            finally:
                _drain_save_futures()
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()

    finally:
        print("正在关闭预处理流水线...")
        pipeline.shutdown()

        print("等待保存任务完成...")
        for fut in save_futures:
            try:
                fut.result(timeout=60)
            except Exception as e:
                print(f"⚠️ 保存异常: {e}")
        save_futures.clear()
        save_executor.shutdown(wait=True)

        gc.collect()
        torch.cuda.empty_cache()
        print("流水线已清理完毕")


def vsibench_aggregate_results(results):
    """聚合评估结果"""
    results_df = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results_df.groupby('question_type').groups.items():
        per_question_type = results_df.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    try:
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy'),
            output.pop('object_rel_direction_medium_accuracy'),
            output.pop('object_rel_direction_hard_accuracy'),
        ]) / 3.
    except:
        output['object_rel_direction_accuracy'] = 0

    output['overall_accuracy'] = sum([_ for _ in output.values()]) / len(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VSI-Bench Evaluation with Qwen3.5 + vLLM')
    parser.add_argument('--parquet_file', type=str, default='./VSI-Bench/test_pruned.parquet')
    parser.add_argument('--video_path', type=str, default='./VSI-Bench')
    parser.add_argument('--res_dir', type=str, default='./results')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--model', type=str, default='/robby/share/Editing/yufengyuan/checkpoint/Qwen3.5-35B-A3B')
    parser.add_argument('--fps', type=float, default=2.0, help='视频采样FPS')
    parser.add_argument('--max_tokens', type=int, default=128, help='最大生成长度')
    parser.add_argument('--tensor_parallel_size', type=int, default=8)
    parser.add_argument('--prefetch_count', type=int, default=16, help='预取视频数量')
    parser.add_argument('--num_workers', type=int, default=8, help='预处理进程数')
    parser.add_argument('--batch_size', type=int, default=8, help='每次推理的样本数')
    parser.add_argument('--enable_thinking', action='store_true', default=False,
                        help='启用 Qwen3.5 思考链')
    args = parser.parse_args()

    vsibench = pd.read_parquet(args.parquet_file)

    if args.save_name:
        res_path = os.path.join(args.res_dir, args.save_name)
    else:
        res_path = os.path.join(args.res_dir, args.model.split('/')[-1])

    os.makedirs(res_path, exist_ok=True)

    # 自动调整 prefetch_count
    effective_prefetch = max(args.prefetch_count, args.batch_size + 1)
    if effective_prefetch != args.prefetch_count:
        print(f"⚠️ prefetch_count 自动调整: {args.prefetch_count} → {effective_prefetch}")

    # 加载 vLLM 模型
    print("正在加载 vLLM 模型...")
    model = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens
    )

    print(f"模型加载完成！")
    print(f"配置: batch_size={args.batch_size}, "
          f"prefetch_count={effective_prefetch}, "
          f"num_workers={args.num_workers}, "
          f"enable_thinking={args.enable_thinking}")

    # 批量处理
    process_vsibench_batch(
        vsibench=vsibench,
        model=model,
        model_path=args.model,
        res_path=res_path,
        video_path_root=args.video_path,
        sampling_params=sampling_params,
        fps=args.fps,
        enable_thinking=args.enable_thinking,
        prefetch_count=effective_prefetch,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # 评估结果
    print("\n正在评估结果...")
    results = []
    with open(os.path.join(res_path, 'response.jsonl'), 'r') as f:
        for line in f:
            doc = json.loads(line)
            processed_doc = vsibench_process_results(doc)
            results.append(processed_doc)

    aggregated_results = vsibench_aggregate_results(results)

    with open(os.path.join(res_path, 'result.json'), "w") as f:
        json.dump(aggregated_results, f, indent=4, ensure_ascii=False)

    print(f"✅ 评估完成！结果已保存至: {res_path}")
    print(f"Overall Accuracy: {aggregated_results['overall_accuracy']:.4f}")