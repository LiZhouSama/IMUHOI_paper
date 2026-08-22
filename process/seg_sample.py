import argparse
import json
import os
import glob
import random
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from tqdm import tqdm

try:
    from process.preprocess import find_unheld_object_displacement_interval
except ImportError:
    from preprocess import find_unheld_object_displacement_interval


def _sequence_length(data: Dict[str, Any]) -> int:
    for key in [
        'position_global_full_gt_world',
        'trans',
        'pose',
    ]:
        value = data.get(key)
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError("无法确定序列长度")


def _is_temporal_value(value: Any, sequence_length: int) -> bool:
    return (
        isinstance(value, (torch.Tensor, np.ndarray))
        and value.ndim >= 1
        and int(value.shape[0]) == int(sequence_length)
    )


def downsample_sequence(data: Dict[str, Any], factor: int = 2) -> Dict[str, Any]:
    """将序列降采样（例如从60fps降到30fps）"""
    sequence_length = _sequence_length(data)
    downsampled = {}
    for key, value in data.items():
        if _is_temporal_value(value, sequence_length):
            downsampled[key] = value[::factor]
        else:
            downsampled[key] = value
    return downsampled


def split_sequence(data: Dict[str, Any], min_len: int, max_len: int) -> List[Dict[str, Any]]:
    """将序列分割为长度在[min_len, max_len]之间的多个子序列"""
    # 获取序列总长度
    T = _sequence_length(data)
    
    segments = []
    start_idx = 0
    
    while start_idx < T:
        # 随机选择当前段的长度
        seg_len = random.randint(min_len, max_len)
        end_idx = start_idx + seg_len
        
        # 如果剩余长度不足min_len，丢弃
        if end_idx > T:
            remaining = T - start_idx
            if remaining < min_len:
                break
            end_idx = T
        
        # 创建子序列
        segment = {}
        for key, value in data.items():
            if _is_temporal_value(value, T):
                segment[key] = value[start_idx:end_idx]
            else:
                segment[key] = value
        
        segments.append(segment)
        start_idx = end_idx
    
    return segments


def find_segment_unheld_object_motion(
    segment: Dict[str, Any],
    displacement_threshold_m: float = 0.10,
) -> Optional[Dict[str, Any]]:
    """Return details when one cropped segment contains substantial unheld motion."""
    if not bool(segment.get("has_object", True)):
        return None

    required_keys = ("obj_trans", "lhand_contact", "rhand_contact")
    missing_keys = [key for key in required_keys if key not in segment]
    if missing_keys:
        raise KeyError(
            "片段级非手交互检测缺少字段: " + ", ".join(missing_keys)
        )

    return find_unheld_object_displacement_interval(
        segment["obj_trans"].float(),
        segment["lhand_contact"].bool(),
        segment["rhand_contact"].bool(),
        min_displacement_m=displacement_threshold_m,
    )


def process_pt_file(input_path: str, output_dir: str, min_len: int, max_len: int, 
                    downsample: bool = False, downsample_factor: int = 2,
                    drop_unheld_object_motion: bool = True,
                    unheld_object_displacement_threshold_m: float = 0.10,
                    skipped_records: Optional[List[Dict[str, Any]]] = None) -> int:
    """处理单个pt文件，返回生成的分割数量"""
    # 加载数据
    data = torch.load(input_path, map_location='cpu')
    seq_name = data.get('seq_name', os.path.splitext(os.path.basename(input_path))[0])
    
    # 降采样（如果需要）
    if downsample:
        data = downsample_sequence(data, factor=downsample_factor)
    
    # 分割序列
    segments = split_sequence(data, min_len, max_len)
    
    # 保存分割后的序列
    os.makedirs(output_dir, exist_ok=True)
    segment_start = 0
    dropped_segments = 0
    for idx, segment in enumerate(segments):
        segment_length = _sequence_length(segment)
        segment_end = segment_start + segment_length - 1
        segment_name = f"{seq_name}_seg{idx:03d}"

        unheld_motion = None
        if drop_unheld_object_motion:
            unheld_motion = find_segment_unheld_object_motion(
                segment,
                displacement_threshold_m=(
                    unheld_object_displacement_threshold_m
                ),
            )
        if unheld_motion is not None:
            dropped_segments += 1
            if skipped_records is not None:
                record = {
                    "seq_name": str(seq_name),
                    "source_file": os.path.basename(input_path),
                    "segment_name": segment_name,
                    "candidate_segment_index": int(idx),
                    "segment_start_frame": int(segment_start),
                    "segment_end_frame": int(segment_end),
                    "segment_frames": int(segment_length),
                    "reason": "object_displacement_without_hand_contact",
                    "lhand_contact_frames": int(
                        segment["lhand_contact"].bool().sum().item()
                    ),
                    "rhand_contact_frames": int(
                        segment["rhand_contact"].bool().sum().item()
                    ),
                    **dict(unheld_motion),
                }
                skipped_records.append(record)
            segment_start = segment_end + 1
            continue

        # 更新seq_name以区分不同分割
        segment['seq_name'] = segment_name
        output_path = os.path.join(output_dir, f"{segment_name}.pt")
        torch.save(segment, output_path)
        segment_start = segment_end + 1
    
    return len(segments) - dropped_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分割和降采样处理后的数据集序列")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="process/processed_split_data_IMHD/train",
        help="输入数据目录，包含待处理的pt文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="process/processed_seg_data_IMHD/train",
        help="输出目录，保存分割后的pt文件"
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=150,
        help="分割后序列的最小长度"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=300,
        help="分割后序列的最大长度"
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="是否进行降采样（例如从60fps降到30fps）"
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=2,
        help="降采样因子，2表示从60fps降到30fps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可重复性"
    )
    drop_group = parser.add_mutually_exclusive_group()
    drop_group.add_argument(
        "--drop-object-displacement-without-hand-contact",
        dest="drop_unheld_object_motion",
        action="store_true",
        help="丢弃裁切后存在无手接触且物体位移超阈值的片段",
    )
    drop_group.add_argument(
        "--keep-object-displacement-without-hand-contact",
        dest="drop_unheld_object_motion",
        action="store_false",
        help="保留裁切后的非手物体运动片段",
    )
    parser.set_defaults(drop_unheld_object_motion=True)
    parser.add_argument(
        "--unheld-object-displacement-threshold-m",
        type=float,
        default=0.10,
        help="片段内无手接触区间允许的最大物体位移（米）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 查找所有pt文件
    pt_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"在 {args.input_dir} 中未找到pt文件")
    
    print(f"找到 {len(pt_files)} 个pt文件")
    print(f"降采样: {'是' if args.downsample else '否'} (因子={args.downsample_factor})")
    print(f"分割长度范围: [{args.min_len}, {args.max_len}]")
    print(
        "片段级非手交互丢弃: "
        f"{'是' if args.drop_unheld_object_motion else '否'} "
        f"(位移阈值={args.unheld_object_displacement_threshold_m:.3f}m)"
    )
    
    total_segments = 0
    skipped_records: List[Dict[str, Any]] = []
    for pt_file in tqdm(pt_files, desc="处理序列"):
        try:
            num_segments = process_pt_file(
                pt_file,
                args.output_dir,
                args.min_len,
                args.max_len,
                args.downsample,
                args.downsample_factor,
                args.drop_unheld_object_motion,
                args.unheld_object_displacement_threshold_m,
                skipped_records,
            )
            total_segments += num_segments
        except Exception as e:
            print(f"\n处理 {os.path.basename(pt_file)} 时出错: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "input_dir": args.input_dir,
        "source_files": len(pt_files),
        "saved_segments": int(total_segments),
        "skipped_segments_count": len(skipped_records),
        "drop_object_displacement_without_hand_contact": bool(
            args.drop_unheld_object_motion
        ),
        "unheld_object_displacement_threshold_m": float(
            args.unheld_object_displacement_threshold_m
        ),
        "segments": skipped_records,
    }
    manifest_path = os.path.join(args.output_dir, "skipped_segments.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    
    print(f"\n完成! 从 {len(pt_files)} 个文件生成了 {total_segments} 个分割片段")
    print(f"片段级非手交互丢弃: {len(skipped_records)} 个")
    print(f"丢弃清单: {manifest_path}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
