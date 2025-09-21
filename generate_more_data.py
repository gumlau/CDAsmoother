#!/usr/bin/env python3
"""
生成更多高质量的训练数据
解决训练太快、数据太少的问题
"""

import os
import h5py
import numpy as np
from cdanet.data import RBDataModule

def generate_large_dataset():
    """生成大型高质量数据集"""
    print("🔄 生成大型训练数据集")
    print("=" * 50)

    data_dir = './rb_data_numerical'
    os.makedirs(data_dir, exist_ok=True)

    # 删除所有现有的数据文件防止覆盖问题
    print("🗑️  清理旧数据文件...")
    for file in os.listdir(data_dir):
        if file.endswith('.h5'):
            old_file_path = os.path.join(data_dir, file)
            print(f"  删除: {file}")
            os.remove(old_file_path)

    # 创建数据模块
    data_module = RBDataModule(
        data_dir=data_dir,
        spatial_downsample=4,
        temporal_downsample=4,
        clip_length=8,
        batch_size=1,
        normalize=True
    )

    # 生成大量数据的数据集
    datasets = [
        {
            'name': 'rb_data_Ra_1e+05.h5',  # 替换现有文件
            'nx': 512,   # 稍微增加分辨率
            'ny': 128,
            'nt': 8000,  # 大幅增加到8000时间步
            'description': '大量训练数据的数据集'
        }
    ]

    generated_files = []

    for i, dataset in enumerate(datasets):
        output_path = os.path.join(data_dir, dataset['name'])

        # 文件已经在上面清理过了，直接生成

        print(f"🔄 生成数据集 {i+1}/{len(datasets)}: {dataset['description']}")
        print(f"   分辨率: {dataset['nx']}x{dataset['ny']}")
        print(f"   时间步: {dataset['nt']}")

        try:
            # 生成数据
            synthetic_file = data_module.create_synthetic_data(
                output_path=output_path,
                Ra=1e5,
                nx=dataset['nx'],
                ny=dataset['ny'],
                nt=dataset['nt']
            )

            # 检查生成的文件
            if os.path.exists(synthetic_file):
                file_size = os.path.getsize(synthetic_file) / 1024 / 1024
                print(f"   ✅ 生成完成: {file_size:.1f} MB")

                # 检查数据质量
                with h5py.File(synthetic_file, 'r') as f:
                    data_shape = f['fields'].shape
                    print(f"   数据形状: {data_shape}")

                generated_files.append(synthetic_file)
            else:
                print(f"   ❌ 生成失败")

        except Exception as e:
            print(f"   ❌ 生成错误: {e}")
            continue

    return generated_files

def estimate_training_samples(data_files):
    """估算训练样本数量"""
    print(f"\n📊 估算训练样本数量:")

    total_clips = 0

    for data_file in data_files:
        if not os.path.exists(data_file):
            continue

        try:
            with h5py.File(data_file, 'r') as f:
                nt = f['fields'].shape[0]  # 时间步数

            # 计算可能的clips数量
            # 每个clip长度为 8 * temporal_downsample = 32 timesteps
            clip_length = 8 * 4  # clip_length * temporal_downsample
            clips_per_file = max(0, nt - clip_length + 1)

            file_size = os.path.getsize(data_file) / 1024 / 1024
            print(f"  {os.path.basename(data_file)}:")
            print(f"    时间步: {nt}")
            print(f"    可能的clips: {clips_per_file}")
            print(f"    文件大小: {file_size:.1f} MB")

            total_clips += clips_per_file

        except Exception as e:
            print(f"  ❌ 无法读取 {data_file}: {e}")

    print(f"\n📈 总计:")
    print(f"  总clips数量: {total_clips}")
    print(f"  按 80/10/10 分割:")
    print(f"    训练集: ~{int(total_clips * 0.8)} samples")
    print(f"    验证集: ~{int(total_clips * 0.1)} samples")
    print(f"    测试集: ~{int(total_clips * 0.1)} samples")

    # 估算训练时间
    if total_clips > 0:
        train_samples = int(total_clips * 0.8)
        epochs_for_1000_samples = 1000 / max(1, train_samples)
        print(f"\n⏱️  训练时间估算:")
        print(f"    每个epoch ~{train_samples} 个样本")

        if train_samples < 100:
            print(f"    ⚠️  仍然太少! 建议生成更多数据")
        elif train_samples < 500:
            print(f"    ⚠️  偏少，但可以尝试")
        else:
            print(f"    ✅ 数量合理")

        if train_samples > 50:
            estimated_time_per_epoch = train_samples * 0.1  # 假设每个样本0.1秒
            print(f"    预计每epoch时间: ~{estimated_time_per_epoch:.1f} 秒")

    return total_clips

def test_data_loading(data_files):
    """测试数据加载"""
    print(f"\n🧪 测试数据加载:")

    try:
        # 创建数据模块
        data_module = RBDataModule(
            data_dir='./rb_data_numerical',
            spatial_downsample=4,
            temporal_downsample=4,
            clip_length=8,
            batch_size=2,  # 测试更大的batch size
            normalize=True,
            num_workers=0
        )

        # 设置数据
        data_module.setup([1e5])

        # 获取数据信息
        data_info = data_module.get_dataset_info()
        print("  数据集信息:")
        for key, info in data_info.items():
            print(f"    {key}: {info['num_samples']} 样本, 形状 {info['high_res_shape']}")

        # 测试数据加载器
        train_loader = data_module.get_dataloader(1e5, 'train')
        test_batch = next(iter(train_loader))

        print(f"  ✅ 数据加载成功")
        print(f"    Batch shapes:")
        print(f"      Low-res: {test_batch['low_res'].shape}")
        print(f"      Targets: {test_batch['targets'].shape}")
        print(f"      Coords: {test_batch['coords'].shape}")

        return True

    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return False

def main():
    print("=" * 60)
    print("高质量训练数据生成器")
    print("=" * 60)

    # 生成大型数据集
    data_files = generate_large_dataset()

    if not data_files:
        print("❌ 没有生成任何数据文件")
        return

    # 估算样本数量
    total_clips = estimate_training_samples(data_files)

    # 测试数据加载
    success = test_data_loading(data_files)

    print(f"\n" + "=" * 60)
    print("📊 数据生成总结:")
    print(f"  生成文件数: {len(data_files)}")
    print(f"  总训练clips: {total_clips}")

    if success and total_clips > 100:
        print("✅ 数据准备完成，可以开始改进训练!")
        print("建议运行:")
        print("  python simple_improved_train.py")
    elif total_clips < 100:
        print("⚠️  数据量仍然不足，建议:")
        print("  1. 增加nt参数 (更多时间步)")
        print("  2. 生成更多数据集")
        print("  3. 减小temporal_downsample")
    else:
        print("❌ 数据加载有问题，需要调试")

    print("=" * 60)

if __name__ == '__main__':
    main()