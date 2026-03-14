#!/bin/bash
# IMUHOI training script (DiT path)

set -e

CONFIG="configs/IMUHOI_train.yaml"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAGE=$1
shift || true

show_help() {
    echo "IMUHOI 训练脚本"
    echo ""
    echo "用法: bash scripts/train.sh <stage> [options]"
    echo ""
    echo "Stage 选项:"
    echo "  hp, human_pose          训练 HumanPoseModule (Stage 1)"
    echo "  int, interaction        训练合并交互模块 (Stage 2/3 merged)"
    echo "  all                     先训练 HP，再训练合并交互模块"
    echo ""
    echo "通用选项:"
    echo "  --cfg FILE              指定配置文件 (默认: $CONFIG)"
    echo "  --debug                 调试模式"
    echo "  --no_trans              使用noTrans模式"
    echo "  --batch_size N          批量大小"
    echo "  --epochs N              训练轮数"
    echo ""
    echo "Interaction 特有选项:"
    echo "  --hp_ckpt PATH          HumanPose权重路径"
    echo "  --interaction_ckpt PATH Interaction权重路径"
    echo "  --freeze_ratio R        冻结阶段比例（默认0.8）"
    echo ""
}

if [ -z "$STAGE" ] || [ "$STAGE" = "-h" ] || [ "$STAGE" = "--help" ]; then
    show_help
    exit 0
fi

case "$STAGE" in
    hp|human_pose)
        echo "=========================================="
        echo "Stage 1: 训练 HumanPoseModule"
        echo "=========================================="
        python train/diffussion/train_human_pose.py --cfg "$CONFIG" "$@"
        ;;

    int|interaction)
        echo "=========================================="
        echo "Stage 2/3: 训练合并交互模块"
        echo "=========================================="
        python train/diffussion/train_interaction.py --cfg "$CONFIG" "$@"
        ;;

    all)
        echo "=========================================="
        echo "依次训练所有阶段"
        echo "=========================================="

        echo ""
        echo "[1/2] 训练 HumanPoseModule..."
        python train/diffussion/train_human_pose.py --cfg "$CONFIG" "$@"

        echo ""
        echo "[2/2] 训练合并交互模块..."
        python train/diffussion/train_interaction.py --cfg "$CONFIG" "$@"

        echo ""
        echo "=========================================="
        echo "所有阶段训练完成!"
        echo "=========================================="
        ;;

    *)
        echo "错误: 未知训练阶段 '$STAGE'"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "训练完成!"
