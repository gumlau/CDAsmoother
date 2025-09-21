#!/usr/bin/env python3
"""
诊断训练后的模型，检查是否真的学到了东西
"""

import torch
import numpy as np
import os
from cdanet.models import CDAnet
from cdanet.data import RBDataModule

def diagnose_model(checkpoint_path):
    """诊断训练后的模型"""
    print("🔍 诊断训练后的模型")
    print("=" * 50)

    # 加载模型
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ 加载模型: {checkpoint_path}")

    # 检查配置
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        print(f"📊 模型配置: {model_config}")
    else:
        model_config = {
            'in_channels': 4, 'feature_channels': 128, 'base_channels': 32,
            'mlp_hidden_dims': [256, 256], 'activation': 'relu',
            'coord_dim': 3, 'output_dim': 4
        }

    # 创建模型
    model = CDAnet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"📈 训练epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"📉 最佳验证loss: {checkpoint.get('best_val_loss', 'Unknown')}")

    # 设置数据
    data_module = RBDataModule(
        data_dir='./rb_data_numerical',
        spatial_downsample=4, temporal_downsample=4,
        batch_size=1, normalize=True, num_workers=0
    )
    data_module.setup([1e5])

    # 获取数据信息
    data_info = data_module.get_dataset_info()
    print(f"\n📊 数据统计:")
    total_samples = 0
    for key, info in data_info.items():
        samples = info['num_samples']
        total_samples += samples
        print(f"  {key}: {samples} 样本")

    print(f"总样本数: {total_samples}")

    if total_samples < 50:
        print("⚠️  警告: 训练数据太少，可能导致快速训练和差的性能")

    # 测试模型预测
    print(f"\n🧪 测试模型预测:")
    test_loader = data_module.get_dataloader(1e5, 'test')
    batch = next(iter(test_loader))

    with torch.no_grad():
        predictions = model(batch['low_res'], batch['coords'])

    # 分析预测
    pred_min = predictions.min().item()
    pred_max = predictions.max().item()
    pred_mean = predictions.mean().item()
    pred_std = predictions.std().item()

    print(f"  预测范围: [{pred_min:.6f}, {pred_max:.6f}]")
    print(f"  预测均值: {pred_mean:.6f}")
    print(f"  预测标准差: {pred_std:.6f}")

    # 分析目标值
    target_min = batch['targets'].min().item()
    target_max = batch['targets'].max().item()
    target_mean = batch['targets'].mean().item()
    target_std = batch['targets'].std().item()

    print(f"  目标范围: [{target_min:.6f}, {target_max:.6f}]")
    print(f"  目标均值: {target_mean:.6f}")
    print(f"  目标标准差: {target_std:.6f}")

    # 判断模型质量
    print(f"\n🎯 诊断结果:")

    if pred_std < 0.001:
        print("❌ 模型预测几乎是常数，没有学到模式")
        print("   建议: 增加训练数据，降低学习率，延长训练时间")
    elif pred_std < 0.01:
        print("⚠️  模型预测变化很小，学习效果差")
        print("   建议: 检查损失函数权重，增加训练时间")
    else:
        print("✅ 模型预测有合理变化")

    # 检查预测和目标的相关性
    pred_flat = predictions.flatten().numpy()
    target_flat = batch['targets'].flatten().numpy()

    # 计算相关系数
    try:
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        print(f"  预测-目标相关系数: {correlation:.4f}")

        if abs(correlation) < 0.1:
            print("❌ 预测和目标几乎无相关性")
        elif abs(correlation) < 0.3:
            print("⚠️  预测和目标相关性较弱")
        else:
            print("✅ 预测和目标有一定相关性")
    except:
        print("⚠️  无法计算相关系数")

    # 检查loss组成
    print(f"\n📈 Loss分析:")

    # 计算回归loss
    mse_loss = torch.nn.functional.mse_loss(predictions, batch['targets'])
    print(f"  MSE Loss: {mse_loss:.6f}")

    # 估算为什么训练这么快
    print(f"\n⏱️  训练速度分析:")
    samples_per_epoch = sum(info['num_samples'] for info in data_info.values())
    print(f"  每个epoch样本数: {samples_per_epoch}")
    print(f"  batch size: 1")
    print(f"  每个epoch的batch数: {samples_per_epoch}")

    if samples_per_epoch < 50:
        print("❌ 每个epoch样本太少，这就是为什么训练这么快!")
        print("   建议:")
        print("   1. 生成更多训练数据 (nx=1024, ny=256, nt=8000+)")
        print("   2. 使用数据增强")
        print("   3. 增加batch size")
        print("   4. 使用更复杂的数据采样策略")

    return predictions, batch['targets']

def suggest_improvements():
    """建议改进方案"""
    print(f"\n🚀 改进建议:")
    print("1. 数据改进:")
    print("   - 生成更高分辨率数据: nx=1024, ny=256")
    print("   - 更多时间步: nt=10000+ (产生更多clips)")
    print("   - 生成多个不同初始条件的数据集")

    print("2. 训练改进:")
    print("   - 降低学习率到 1e-4 或更小")
    print("   - 增加PDE loss权重到 0.01")
    print("   - 延长训练到 2000+ epochs")
    print("   - 使用学习率调度器")

    print("3. 模型改进:")
    print("   - 增加模型复杂度 (更多特征通道)")
    print("   - 添加dropout防止过拟合")
    print("   - 使用残差连接")

def main():
    checkpoint_path = './checkpoints/improved_model_final.pth'

    if os.path.exists(checkpoint_path):
        diagnose_model(checkpoint_path)
    else:
        print(f"❌ 找不到模型文件: {checkpoint_path}")
        print("请先运行训练脚本")

    suggest_improvements()

if __name__ == '__main__':
    main()