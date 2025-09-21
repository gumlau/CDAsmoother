#!/usr/bin/env python3
"""
改进的训练脚本，解决训练太快和效果差的问题
"""

import os
import torch
from datetime import datetime

from cdanet.models import CDAnet
from cdanet.config import ExperimentConfig
from cdanet.data import RBDataModule
from cdanet.training import CDAnetTrainer
from cdanet.utils import Logger


def main():
    print("=" * 60)
    print("CDAnet 改进训练")
    print("=" * 60)

    # 创建配置
    config = ExperimentConfig()

    config.experiment_name = f"cdanet_better_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.description = "改进的训练配置，更多数据和更好参数"

    # 数据配置 - 减少downsampling增加数据量
    config.data.data_dir = os.path.abspath('./rb_data_numerical')
    config.data.spatial_downsample = 2  # 从4减少到2，增加4倍空间数据
    config.data.temporal_downsample = 2  # 从4减少到2，增加4倍时间数据
    config.data.clip_length = 16  # 增加clip长度
    config.data.Ra_numbers = [1e5]
    config.data.batch_size = 1  # 减小batch size，更多步数
    config.data.num_workers = 0
    config.data.pde_points = 4096  # 增加PDE点
    config.data.normalize = True

    # 模型配置 - 和checkpoint匹配
    config.model.in_channels = 4
    config.model.feature_channels = 128
    config.model.base_channels = 32
    config.model.mlp_hidden_dims = [256, 256]
    config.model.activation = 'relu'
    config.model.coord_dim = 3
    config.model.output_dim = 4

    # 损失配置 - 几乎关闭PDE loss专注数据拟合
    config.loss.lambda_pde = 0.000001  # 极小的PDE权重
    config.loss.regression_norm = 'l2'
    config.loss.pde_norm = 'l2'
    config.loss.Ra = 1e5
    config.loss.Pr = 0.7
    config.loss.Lx = 3.0
    config.loss.Ly = 1.0

    # 优化器配置 - 更激进的学习
    config.optimizer.optimizer_type = 'adam'
    config.optimizer.learning_rate = 0.005  # 更高的学习率
    config.optimizer.weight_decay = 1e-6    # 更小的weight decay
    config.optimizer.grad_clip_max_norm = 2.0  # 放松梯度裁剪
    config.optimizer.scheduler_type = 'step'   # 使用step scheduler
    config.optimizer.step_size = 50           # 每50 epochs降低LR
    config.optimizer.gamma = 0.8               # LR衰减因子

    # 训练配置 - 更仔细的训练
    config.training.num_epochs = 200  # 减少epochs，关注质量
    config.training.clips_per_epoch = -1
    config.training.val_interval = 5   # 更频繁验证
    config.training.checkpoint_interval = 25
    config.training.save_best = True
    config.training.early_stopping = True
    config.training.patience = 50      # 减少patience
    config.training.min_delta = 1e-4   # 更大的min_delta要求真正的改进
    config.training.use_amp = False
    config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.training.log_interval = 1
    config.training.output_dir = './outputs'
    config.training.checkpoint_dir = './checkpoints'
    config.training.log_dir = './logs'

    print(f"实验名称: {config.experiment_name}")
    print(f"学习率: {config.optimizer.learning_rate}")
    print(f"PDE权重: {config.loss.lambda_pde}")
    print(f"设备: {config.training.device}")

    # 强制重新生成干净的数据
    print("🗑️  删除有问题的旧数据，重新生成...")

    data_file = os.path.join(config.data.data_dir, f'rb_data_Ra_{config.data.Ra_numbers[0]:.0e}.h5')
    if os.path.exists(data_file):
        os.remove(data_file)
        print("  删除旧数据文件")

    # 使用改进的RB simulation生成更多真实数据
    print("🔄 运行改进的RB simulation...")
    import subprocess
    result = subprocess.run([
        'python3', 'rb_simulation.py',
        '--Ra', '1e5',
        '--n_runs', '20',  # 20个runs
        '--nx', '512',     # 更高分辨率
        '--ny', '128',
        '--nt', '3000'     # 更多时间步
    ], capture_output=True, text=True, cwd='.')

    if result.returncode == 0:
        print("✅ RB simulation完成")
        # 转换数据
        conv_result = subprocess.run(['python3', 'convert_rb_data.py'], capture_output=True, text=True)
        if conv_result.returncode == 0:
            print("✅ 数据转换完成")
        else:
            print("⚠️  数据转换有问题，但继续训练")
    else:
        print(f"❌ RB simulation失败: {result.stderr}")
        print("程序终止")
        return

    print("=" * 60)

    # 设置数据
    data_module = RBDataModule(
        data_dir=config.data.data_dir,
        spatial_downsample=config.data.spatial_downsample,
        temporal_downsample=config.data.temporal_downsample,
        clip_length=config.data.clip_length,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pde_points=config.data.pde_points,
        normalize=config.data.normalize
    )

    data_module.setup(config.data.Ra_numbers)

    # 显示数据统计
    data_info = data_module.get_dataset_info()
    total_samples = sum(info['num_samples'] for info in data_info.values())
    print(f"📊 总样本数: {total_samples}")

    for key, info in data_info.items():
        print(f"  {key}: {info['num_samples']} samples")

    if total_samples < 100:
        print("⚠️  样本数仍然太少，训练可能很快完成")

    # 创建模型
    model = CDAnet(
        in_channels=config.model.in_channels,
        feature_channels=config.model.feature_channels,
        base_channels=config.model.base_channels,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        activation=config.model.activation,
        coord_dim=config.model.coord_dim,
        output_dim=config.model.output_dim
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数: {total_params:,}")

    # 设置日志
    logger = Logger(
        log_dir=config.training.log_dir,
        experiment_name=config.experiment_name,
        use_tensorboard=True,
        use_wandb=False,
        config=config.to_dict()
    )

    # 创建训练器
    trainer = CDAnetTrainer(
        config=config,
        model=model,
        data_module=data_module,
        logger=logger
    )

    # 开始训练
    print("🚀 开始训练...")
    try:
        trainer.train()
        print("✅ 训练完成")

        print(f"📊 最终统计:")
        print(f"  最终epoch: {trainer.current_epoch}")
        print(f"  最佳验证loss: {trainer.best_val_loss:.6f}")

    except KeyboardInterrupt:
        print("⏹️  用户中断训练")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise
    finally:
        logger.close()

    # 保存模型
    final_checkpoint = os.path.join(config.training.checkpoint_dir, 'better_model_final.pth')
    try:
        trainer._save_checkpoint('better_model_final.pth', trainer.current_epoch, {})
        print(f"💾 模型已保存: {final_checkpoint}")
    except Exception as e:
        print(f"⚠️  保存失败: {e}")

    print("=" * 60)
    print("训练完成! 运行以下命令查看结果:")
    print(f"python visualize_results.py --checkpoint {final_checkpoint}")
    print("=" * 60)


if __name__ == '__main__':
    main()