#!/usr/bin/env python3
"""
跨平台训练启动脚本
自动检测环境并使用最佳设置
"""

import os
import sys
import torch
import platform
import subprocess


def detect_environment():
    """检测运行环境并返回最佳配置"""
    system = platform.system()
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    print(f"🔍 检测到系统: {system}")
    print(f"🔍 CUDA可用: {cuda_available}")
    print(f"🔍 MPS可用: {mps_available}")

    if cuda_available:
        return {
            'device': 'cuda',
            'batch_size': 4,
            'nx': 128,
            'nz': 64,
            'nt': 16,
            'workers': 8,
            'description': '🚀 CUDA GPU训练'
        }
    elif mps_available:
        # MPS有一些限制，使用CPU代替某些操作
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        return {
            'device': 'mps',
            'batch_size': 2,
            'nx': 64,
            'nz': 32,
            'nt': 8,
            'workers': 4,
            'description': '🍎 Apple Silicon MPS训练'
        }
    else:
        return {
            'device': 'cpu',
            'batch_size': 2,
            'nx': 64,
            'nz': 32,
            'nt': 8,
            'workers': 4,
            'description': '💻 CPU训练'
        }


def check_data():
    """检查数据文件"""
    data_file = 'rb_data_numerical/rb2d_ra1e+05_consolidated.h5'
    if os.path.exists(data_file):
        print("✅ 训练数据已准备好")
        return True
    else:
        print("❌ 训练数据不存在")
        print("请先运行: python3 convert_existing_data.py")
        return False


def start_training(config, epochs=50, quick_test=False):
    """启动训练"""
    if quick_test:
        epochs = 5
        config['batch_size'] = min(config['batch_size'], 2)
        print(f"🧪 快速测试模式: {epochs} epochs")

    print(f"\n{config['description']}")
    print(f"设备: {config['device']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"网格大小: {config['nx']}×{config['nz']}×{config['nt']}")

    # 构建训练命令
    cmd = [
        sys.executable, 'train_cdanet.py',
        '--device', config['device'],
        '--batch_size', str(config['batch_size']),
        '--epochs', str(epochs),
        '--nx', str(config['nx']),
        '--nz', str(config['nz']),
        '--nt', str(config['nt']),
        '--num_workers', str(config['workers']),
        '--lr', '0.01',
        '--alpha_pde', '0.01',
        '--output_folder', f'./outputs_{config["device"]}'
    ]

    if quick_test:
        cmd.extend(['--pseudo_epoch_size', '100'])

    print(f"\n执行命令:")
    print(' '.join(cmd))
    print(f"\n{'='*60}")

    # 启动训练
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 训练完成！")
        print(f"📁 检查点保存在: ./outputs_{config['device']}/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ 训练被用户中断")
        return False

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='跨平台CDAnet训练启动器')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    parser.add_argument('--force-device', type=str, choices=['cuda', 'mps', 'cpu'],
                       help='强制指定设备')

    args = parser.parse_args()

    print("🚀 CDAnet 跨平台训练启动器")
    print("=" * 60)

    # 检查数据
    if not check_data():
        return

    # 检测环境
    if args.force_device:
        config = {
            'device': args.force_device,
            'batch_size': 2 if args.force_device != 'cuda' else 4,
            'nx': 64 if args.force_device != 'cuda' else 128,
            'nz': 32 if args.force_device != 'cuda' else 64,
            'nt': 8 if args.force_device != 'cuda' else 16,
            'workers': 4 if args.force_device != 'cuda' else 8,
            'description': f'🔧 强制使用 {args.force_device}'
        }
    else:
        config = detect_environment()

    # 启动训练
    success = start_training(config, args.epochs, args.quick)

    if success:
        print("\n🎉 训练任务完成！")
        print("\n📊 后续步骤:")
        print("1. 查看训练日志:")
        print(f"   tensorboard --logdir outputs_{config['device']}/tensorboard")
        print("2. 评估模型:")
        print("   python3 evaluate_cdanet.py")
        print("3. 可视化结果:")
        print("   python3 visualize_results.py")
    else:
        print("\n💡 故障排除建议:")
        print("1. 检查设备兼容性: python3 test_compatibility.py")
        print("2. 尝试CPU模式: python3 start_training.py --force-device cpu")
        print("3. 快速测试: python3 start_training.py --quick")


if __name__ == '__main__':
    main()