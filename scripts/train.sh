#!/bin/bash
# IMUHOI 训练脚本
# 
# 用法示例:
#   # Stage 1: 训练 VelocityContact
#   bash scripts/train.sh vc
#   
#   # Stage 2: 训练 HumanPose (普通模式)
#   bash scripts/train.sh hp
#   
#   # Stage 2: 训练 HumanPose (noTrans模式)
#   bash scripts/train.sh hp --no_trans
#   
#   # Stage 3: 训练 ObjectTrans
#   bash scripts/train.sh ot --vc_ckpt path/to/vc.pt --hp_ckpt path/to/hp.pt
#   
#   # Stage 3: 自动搜索权重
#   bash scripts/train.sh ot --auto_find_ckpt
#   
#   # 联合训练所有模块
#   bash scripts/train.sh joint --auto_find_ckpt
#   
#   # 调试模式
#   bash scripts/train.sh vc --debug

set -e

# 默认配置文件
CONFIG="configs/IMUHOI_train.yaml"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 解析第一个参数作为训练阶段
STAGE=$1
shift || true

# 显示帮助
show_help() {
    echo "IMUHOI 训练脚本"
    echo ""
    echo "用法: bash scripts/train.sh <stage> [options]"
    echo ""
    echo "Stage 选项:"
    echo "  vc, velocity_contact    训练 VelocityContactModule (Stage 1)"
    echo "  hp, human_pose          训练 HumanPoseModule (Stage 2)"  
    echo "  ot, object_trans        训练 ObjectTransModule (Stage 3)"
    echo "  joint                   联合训练所有模块"
    echo "  all                     依次训练所有阶段"
    echo ""
    echo "通用选项:"
    echo "  --cfg FILE              指定配置文件 (默认: $CONFIG)"
    echo "  --debug                 调试模式 (小数据集, 少epoch)"
    echo "  --no_trans              使用noTrans模式 (Stage 2+)"
    echo "  --batch_size N          批量大小"
    echo "  --epochs N              训练轮数"
    echo ""
    echo "Stage 3 / Joint 特有选项:"
    echo "  --vc_ckpt PATH          VelocityContact权重路径"
    echo "  --hp_ckpt PATH          HumanPose权重路径"
    echo "  --auto_find_ckpt        自动搜索最新权重"
    echo ""
    echo "示例:"
    echo "  bash scripts/train.sh vc                              # 训练Stage1"
    echo "  bash scripts/train.sh hp --no_trans                   # 训练Stage2 noTrans模式"
    echo "  bash scripts/train.sh ot --auto_find_ckpt             # 训练Stage3 自动搜索权重"
    echo "  bash scripts/train.sh all --no_trans                  # 依次训练所有阶段"
    echo "  bash scripts/train.sh vc --debug --batch_size 4       # 调试模式"
}

# 检查stage参数
if [ -z "$STAGE" ] || [ "$STAGE" = "-h" ] || [ "$STAGE" = "--help" ]; then
    show_help
    exit 0
fi

# 运行训练
case "$STAGE" in
    vc|velocity_contact)
        echo "=========================================="
        echo "Stage 1: 训练 VelocityContactModule"
        echo "=========================================="
        python train/train_velocity_contact.py --cfg "$CONFIG" "$@"
        ;;
    
    hp|human_pose)
        echo "=========================================="
        echo "Stage 2: 训练 HumanPoseModule"
        echo "=========================================="
        python train/train_human_pose.py --cfg "$CONFIG" "$@"
        ;;
    
    ot|object_trans)
        echo "=========================================="
        echo "Stage 3: 训练 ObjectTransModule"
        echo "=========================================="
        python train/train_object_trans.py --cfg "$CONFIG" "$@"
        ;;
    
    joint)
        echo "=========================================="
        echo "联合训练所有模块"
        echo "=========================================="
        python train/train_object_trans.py --cfg "$CONFIG" --joint_train "$@"
        ;;
    
    all)
        echo "=========================================="
        echo "依次训练所有阶段"
        echo "=========================================="
        
        echo ""
        echo "[1/3] 训练 VelocityContactModule..."
        python train/train_velocity_contact.py --cfg "$CONFIG" "$@"
        
        echo ""
        echo "[2/3] 训练 HumanPoseModule..."
        python train/train_human_pose.py --cfg "$CONFIG" "$@"
        
        echo ""
        echo "[3/3] 训练 ObjectTransModule..."
        python train/train_object_trans.py --cfg "$CONFIG" --auto_find_ckpt "$@"
        
        echo ""
        echo "=========================================="
        echo "所有阶段训练完成!"
        echo "=========================================="
        ;;
    
    *)
        echo "错误: 未知的训练阶段 '$STAGE'"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "训练完成!"

