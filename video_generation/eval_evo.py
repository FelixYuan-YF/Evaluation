import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot


def get_args_parser():
    parser = argparse.ArgumentParser()
    # 核心参数：指定pred和gt的npy文件路径（必填）
    parser.add_argument(
        "--pred_npy",
        type=str,
        required=True,
        help="path to prediction npy file (Realestimate10K format)"
    )
    parser.add_argument(
        "--gt_npy",
        type=str,
        required=True,
        help="path to ground truth npy file (Realestimate10K format)"
    )
    # 可选：位姿采样步长（保留，若需要下采样）
    parser.add_argument(
        "--pose_eval_stride",
        default=1,
        type=int,
        help="stride for pose evaluation (default: 1)"
    )
    # 轨迹图保存路径（可选，默认保存到当前目录的trajectory.png）
    parser.add_argument(
        "--traj_plot_path",
        type=str,
        default="./trajectory.png",
        help="path to save trajectory comparison plot (default: ./trajectory.png)"
    )
    # 日志文件路径（可选，保存评估指标文本）
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="path to save evaluation metrics log (default: None, do not save)"
    )
    return parser


def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


def to_numpy(x):
    return todevice(x, "numpy")


def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation

    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = to_numpy(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose


def get_tum_poses(poses):
    """
    poses: list of 4x4 arrays
    """
    tt = np.arange(len(poses)).astype(float)
    tum_poses = [c2w_to_tumpose(p) for p in poses]
    tum_poses = np.stack(tum_poses, 0)
    return [tum_poses, tt]


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)


def plot_trajectory(
    pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True
):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("traj_error", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved trajectory to {filename.replace('.png','')}_traj_error.png")


def eval_metrics(pred_traj, gt_traj=None, seq="", filename="", sample_stride=1):

    if sample_stride > 1:
        pred_traj[0] = pred_traj[0][::sample_stride]
        pred_traj[1] = pred_traj[1][::sample_stride]
        if gt_traj is not None:
            updated_gt_traj = []
            updated_gt_traj.append(gt_traj[0][::sample_stride])
            updated_gt_traj.append(gt_traj[1][::sample_stride])
            gt_traj = updated_gt_traj

    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)

        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    # ATE
    traj_ref = gt_traj
    traj_est = pred_traj

    ate_result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )

    ate = ate_result.stats["rmse"]
    # print(ate_result.np_arrays['error_array'])
    # exit()

    # RPE rotation and translation
    delta_list = [1]
    rpe_rots, rpe_transs = [], []
    for delta in delta_list:
        rpe_rots_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        rot = rpe_rots_result.stats["rmse"]
        rpe_rots.append(rot)

    for delta in delta_list:
        rpe_transs_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        trans = rpe_transs_result.stats["rmse"]
        rpe_transs.append(trans)

    rpe_trans, rpe_rot = np.mean(rpe_transs), np.mean(rpe_rots)
    if filename is not None:
        with open(filename, "w+") as f:
            f.write(f"Seq: {seq} \n\n")
            f.write(f"{ate_result}")
            f.write(f"{rpe_rots_result}")
            f.write(f"{rpe_transs_result}")

        print(f"Save results to {filename}")
    return ate, rpe_trans, rpe_rot


def parse_realestimate10k_npy(npy_path):
    """
    解析Realestimate10K格式的npy文件
    格式：每个条目为 [i, fx, fy, cx, cy, 0, 0, p1, p2, ..., p12]
    参数：
        npy_path: npy文件路径
    返回：
        indices: 帧索引列表（已排序）
        poses: 4x4齐次位姿矩阵列表（numpy数组，已排序）
    """
    # 加载npy文件（支持任意维度的npy，只要每个元素符合格式）
    data = np.load(npy_path, allow_pickle=True)
    # 处理一维数组的情况（若npy是一维的，每个元素是单条数据）
    if data.ndim == 1 and len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
        data = data.reshape(-1, len(data[0]))
    # 确保是二维数组（行：帧，列：格式字段）
    if data.ndim != 2:
        raise ValueError(f"Invalid npy format: expected 2D array, got {data.ndim}D")
    # 检查每一行的长度是否符合要求（至少19个元素：0-18索引）
    if data.shape[1] < 19:
        raise ValueError(
            f"Invalid npy format: each row must have at least 19 elements, got {data.shape[1]}")

    indices = []
    poses = []

    for entry in data:
        # 解析基础信息
        i = int(entry[0])
        # 跳过第6、7位（0,0），提取p1-p12（共12个元素，索引7-18）
        p_12 = entry[7:19]
        # 重塑为3x4的RT矩阵（R: 3x3, t: 3x1）
        rt_matrix = p_12.reshape(3, 4)
        # 构建4x4齐次位姿矩阵（相机位姿：相机→世界 或 世界→相机，需与GT一致）
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :4] = rt_matrix  # 前3行是RT，最后一行是[0,0,0,1]
        # 求逆得到c2w矩阵
        pose_4x4 = np.linalg.inv(pose_4x4)
        # 存入列表
        indices.append(i)
        poses.append(pose_4x4)

    # 按帧索引排序（防止npy中顺序混乱）
    sorted_idx = np.argsort(indices)
    indices = [indices[j] for j in sorted_idx]
    poses = [poses[j] for j in sorted_idx]

    return indices, poses


def eval_single_pose_pair(args):
    """
    处理单个pred和gt npy文件，计算评估指标并绘制轨迹图（仅保留分数输出和轨迹图）
    """
    try:
        # --------------------------
        # 1. 解析pred和gt的npy文件（仅提取位姿）
        # --------------------------
        print(f"Parsing prediction npy: {args.pred_npy}")
        pred_indices, pred_poses_np = parse_realestimate10k_npy(args.pred_npy)
        print(f"Parsing ground truth npy: {args.gt_npy}")
        gt_indices, gt_poses_np = parse_realestimate10k_npy(args.gt_npy)

        # --------------------------
        # 2. 按步长采样（可选）
        # --------------------------
        stride = args.pose_eval_stride
        pred_poses_np = np.array(pred_poses_np)[::stride]
        gt_poses_np = np.array(gt_poses_np)[::stride]

        # 检查pred和gt的长度是否匹配
        if len(pred_poses_np) != len(gt_poses_np):
            print(f"Warning: pred length ({len(pred_poses_np)}) != gt length ({len(gt_poses_np)})")
            # 取较短的长度（可选：用户可根据需求调整）
            min_len = min(len(pred_poses_np), len(gt_poses_np))
            pred_poses_np = pred_poses_np[:min_len]
            gt_poses_np = gt_poses_np[:min_len]
            print(f"Auto-truncated to min length: {min_len}")

        # --------------------------
        # 3. 格式转换：适配评估函数（转为torch张量）
        # --------------------------
        pr_poses = torch.tensor(pred_poses_np, dtype=torch.float32)
        gt_poses = torch.tensor(gt_poses_np, dtype=torch.float32)

        # 转换为TUM格式轨迹（供eval_metrics计算指标、plot_trajectory绘图）
        pred_traj = get_tum_poses(pr_poses)
        gt_traj = get_tum_poses(gt_poses)

        # --------------------------
        # 4. 计算评估指标（ATE、RPE）
        # --------------------------
        print("Calculating evaluation metrics (ATE/RPE)...")
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj,
            seq="single_pair",  # 序列名（无实际意义，仅用于内部日志）
            filename=args.log_file
        )

        # --------------------------
        # 5. 绘制并保存轨迹对比图
        # --------------------------
        print(f"Plotting trajectory comparison, saving to: {args.traj_plot_path}")
        # 创建轨迹图所在的目录（若不存在）
        plot_dir = os.path.dirname(args.traj_plot_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # 绘制轨迹图
        plot_trajectory(
            pred_traj,
            gt_traj,
            title="Pred vs GT Trajectory",
            filename=args.traj_plot_path
        )

        # --------------------------
        # 6. 输出结果（仅打印分数）
        # --------------------------
        print("\n===== Final Evaluation Results =====")
        print(f"ATE (Absolute Trajectory Error): {ate:.5f}")
        print(f"RPE Translation (Relative Pose Error - Translation): {rpe_trans:.5f}")
        print(f"RPE Rotation (Relative Pose Error - Rotation): {rpe_rot:.5f}")

        return ate, rpe_trans, rpe_rot

    except Exception as e:
        # 异常处理：打印错误信息并抛出
        print(f"Error during evaluation: {str(e)}")
        raise e


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    # 执行评估
    eval_single_pose_pair(args)
