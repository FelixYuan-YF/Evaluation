import argparse
import torch
from torch import Tensor
import pandas as pd
import numpy as np
from typing import Literal

def rt34_to_44(rt: Tensor) -> Tensor:
    dummy = torch.tensor([[[0, 0, 0, 1]]] * rt.size(0), dtype=rt.dtype, device=rt.device)
    return torch.cat([rt, dummy], dim=1)


def relative_pose(rt: Tensor, mode: Literal["left", "right"]) -> Tensor:
    if mode == "left":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[:1].inverse() @ rt[1:]], dim=0)
    elif mode == "right":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[1:] @ rt[:1].inverse()], dim=0)
    return rt


def normalize_t(rt: Tensor, ref: Tensor = None, eps: float = 1e-9):
    if ref is None:
        ref = rt
    scale = ref[:, :3, 3:4].norm(p=2, dim=1).amax() + eps
    return rt34_to_44(torch.cat([rt[:, :3, :3], rt[:, :3, 3:4] / scale], dim=-1))

# -------------------------- 1. 整合你的姿态转换函数（核心！保持与你的逻辑一致） --------------------------
def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """Convert 4-dimensional quaternions to 3x3 rotation matrices（复用你的实现）"""
    i = quaternions[..., 0]  # qx
    j = quaternions[..., 1]  # qy
    k = quaternions[..., 2]  # qz
    r = quaternions[..., 3]  # qw
    
    two_s = 2.0 / ((quaternions **2).sum(axis=-1) + eps)
    
    o = np.stack([
        1 - two_s * (j** 2 + k **2),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i** 2 + k **2),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i** 2 + j ** 2)
    ], axis=-1)
    
    return o.reshape(*o.shape[:-1], 3, 3)


def pose_from_quaternion(pose):
    """Convert [tx,ty,tz,qx,qy,qz,qw] to 3x4 w2c matrix（复用你的实现）"""
    if not isinstance(pose, np.ndarray):
        pose = np.array(pose)
    
    # 处理1D单个pose（如(7,)），自动加batch维度后再转换
    if len(pose.shape) == 1:
        pose = pose[np.newaxis, :]
    
    quat_t = pose[..., :3]  # 平移分量（w2c的tx, ty, tz）
    quat_r = pose[..., 3:]  # 四元数旋转分量（qx, qy, qz, qw）
    
    w2c_matrix = np.zeros((*pose.shape[:-1], 3, 4), dtype=pose.dtype)
    w2c_matrix[..., :3, 3] = quat_t  # 平移部分
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)  # 旋转部分
    
    return w2c_matrix


# -------------------------- 2. 核心保留：误差计算函数 --------------------------
def calc_roterr(r1: Tensor, r2: Tensor) -> Tensor:  # 旋转误差（弧度）
    return (((r1.transpose(-1, -2) @ r2).diagonal(dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1).acos()


def calc_transerr(t1: Tensor, t2: Tensor) -> Tensor:  # 平移误差（L2范数）
    return (t2 - t1).norm(p=2, dim=-1)


def calc_cammc(rt1: Tensor, rt2: Tensor) -> Tensor:  # 姿态矩阵整体误差（3×4展平后L2）
    return (rt2 - rt1).reshape(-1, 12).norm(p=2, dim=-1)


def metric(c2w_1: Tensor, c2w_2: Tensor) -> tuple[float, float, float, float, float]:
    """核心误差计算：输入两个(n,3,4)的c2w相对姿态，输出5个误差指标"""
    # 1. 旋转误差
    RotErr = calc_roterr(c2w_1[:, :3, :3], c2w_2[:, :3, :3]).sum().item()

    # 2. 相对尺度误差（以各自为基准）
    c2w_1_rel = normalize_t(c2w_1, c2w_1)
    c2w_2_rel = normalize_t(c2w_2, c2w_2)
    TransErr_rel = calc_transerr(c2w_1_rel[:, :3, 3], c2w_2_rel[:, :3, 3]).sum().item()
    CamMC_rel = calc_cammc(c2w_1_rel[:, :3, :4], c2w_2_rel[:, :3, :4]).sum().item()

    # 3. 绝对尺度误差（以GT为基准）
    c2w_1_abs = normalize_t(c2w_1, c2w_1)
    c2w_2_abs = normalize_t(c2w_2, c2w_1)
    TransErr_abs = calc_transerr(c2w_1_abs[:, :3, 3], c2w_2_abs[:, :3, 3]).sum().item()
    CamMC_abs = calc_cammc(c2w_1_abs[:, :3, :4], c2w_2_abs[:, :3, :4]).sum().item()

    return RotErr, TransErr_rel, CamMC_rel, TransErr_abs, CamMC_abs


# -------------------------- 3. 适配你的npy格式：加载并转换为相对c2w姿态 --------------------------
def load_npy_and_convert_pose(
    poses_npy_path: str
) -> Tensor:
    """
    读取你的poses.npy和intrinsics.npy，转换为(n,3,4)的相对c2w姿态
    完全遵循你的转换逻辑：w2c → 4x4 → 求逆 → c2w → 计算相对姿态
    """
    # 1. 读取npy文件
    poses_np = np.load(poses_npy_path)  # (n,7)：[tx, ty, tz, qx, qy, qz, qw]（w2c的参数）

    # 2. 校验维度
    assert poses_np.shape[1] == 7, f"poses必须是(n,7)，当前为{poses_np.shape}"

    # 3. 转换为w2c矩阵（3x4）→ 扩展为4x4 → 求逆得到c2w矩阵（4x4）
    n_frames = poses_np.shape[0]
    c2w_list = []
    for i in range(n_frames):
        # 3.1 提取单帧姿态，转换为3x4 w2c矩阵（复用你的pose_from_quaternion）
        single_w2c_3x4 = pose_from_quaternion(poses_np[i]).squeeze(0)  # (3,4)
        # 3.2 扩展为4x4 w2c矩阵（添加齐次行[0,0,0,1]）
        single_w2c_4x4 = np.vstack([single_w2c_3x4, [0, 0, 0, 1]])  # (4,4)
        # 3.3 求逆得到c2w矩阵（4x4）（完全遵循你的load_extrinsics逻辑）
        single_c2w_4x4 = np.linalg.inv(single_w2c_4x4)  # (4,4)
        # 3.4 保存
        c2w_list.append(single_c2w_4x4)  # (4,4)
    
    # 4. 拼接所有帧的c2w矩阵，转换为tensor
    c2w_mat = torch.tensor(np.stack(c2w_list, axis=0), dtype=torch.float32)  # (n,4,4)

    # 5. 计算相对姿态（以首帧为基准）
    rel_c2w = relative_pose(c2w_mat, mode="left")
    return rel_c2w


# -------------------------- 4. 主函数：读取npy→转换→计算误差→输出 --------------------------
def main():
    parser = argparse.ArgumentParser(description="使用你的姿态转换逻辑，评估npy格式的GT与估计结果")
    # 输入路径
    parser.add_argument("--csv_path", type=str, help="包含GT与估计结果路径的CSV文件（可选）")
    parser.add_argument("--dir_path", type=str, help="EST姿态npy文件路径")
    # 输出路径
    parser.add_argument("--output_csv", type=str, default="eval.csv", help="误差输出CSV路径")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)

    metric_names = ["RotErr", "TransErr_rel", "CamMC_rel", "TransErr_abs", "CamMC_abs"]
    
    for idx, row in df.iterrows():
        gt_path = row['pose_path']
        est_path = f"{args.dir_path}/{row['id']}/reconstructions/poses.npy"

        # 1. 读取并转换GT和估计的姿态（完全用你的转换逻辑）
        gt_rel_c2w = load_npy_and_convert_pose(
            poses_npy_path=gt_path
        )[:17]
        est_rel_c2w = load_npy_and_convert_pose(
            poses_npy_path=est_path
        )

        # 2. 校验帧数量一致
        if gt_rel_c2w.shape[0] != est_rel_c2w.shape[0]:
            print(f"GT与估计帧数不匹配！GT:{gt_rel_c2w.shape[0]}, Est:{est_rel_c2w.shape[0]}")
            return

        # 3. 计算误差并输出
        metrics = metric(gt_rel_c2w, est_rel_c2w)

        # 保存到CSV
        results = {name: value for name, value in zip(metric_names, metrics)}
        for name, value in results.items():
            df.at[idx, name] = value

    df.to_csv(args.output_csv, index=False)
    print(f"评估结果已保存到 {args.output_csv}")


if __name__ == "__main__":
    main()