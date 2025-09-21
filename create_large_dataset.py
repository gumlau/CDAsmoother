#!/usr/bin/env python3
"""
创建大型训练数据集，解决训练数据不足问题
"""

import os
import h5py
import numpy as np

def create_large_rb_dataset():
    """创建大型RB数据集"""
    print("🔄 创建大型RB训练数据集")
    print("=" * 50)

    data_dir = './rb_data_numerical'
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, 'rb_data_Ra_1e+05.h5')

    # 删除旧文件
    if os.path.exists(output_file):
        print(f"🗑️  删除旧文件: {output_file}")
        os.remove(output_file)

    # 参数设置
    Ra = 1e5
    Pr = 0.7
    Lx, Ly = 3.0, 1.0

    # 高分辨率网格
    nx, ny = 768, 192  # 高分辨率
    nt = 6000          # 更多时间步

    print(f"📊 数据集参数:")
    print(f"  分辨率: {nx} x {ny}")
    print(f"  时间步: {nt}")
    print(f"  Ra数: {Ra}")

    # 创建坐标网格
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    print("🔄 生成合成数据...")

    # 初始化场
    fields = np.zeros((nt, nx, ny, 4))  # [time, x, y, channels]

    # 时间参数
    dt = 0.01
    t_values = np.arange(nt) * dt

    for t_idx, t in enumerate(t_values):
        if t_idx % 1000 == 0:
            print(f"  进度: {t_idx}/{nt} ({100*t_idx/nt:.1f}%)")

        # 生成更复杂的温度场模式
        # 主要对流模式
        T_base = 0.5 + 0.3 * np.sin(2*np.pi*X/Lx) * np.sin(np.pi*Y/Ly)

        # 添加时间演化的扰动
        T_perturbation = (
            0.1 * np.sin(4*np.pi*X/Lx + 0.1*t) * np.cos(2*np.pi*Y/Ly) +
            0.05 * np.cos(6*np.pi*X/Lx - 0.05*t) * np.sin(3*np.pi*Y/Ly) +
            0.02 * np.sin(8*np.pi*X/Lx + 0.03*t) * np.sin(4*np.pi*Y/Ly)
        )

        # 小尺度湍流扰动
        turbulent_noise = 0.01 * np.random.normal(0, 1, (nx, ny))

        # 边界层效应
        boundary_effect = 0.1 * np.exp(-10*Y/Ly) * np.sin(4*np.pi*X/Lx + 0.2*t)

        T = T_base + T_perturbation + turbulent_noise + boundary_effect

        # 速度场（简化的对流模式）
        u = 0.2 * np.sin(2*np.pi*Y/Ly) * np.cos(2*np.pi*X/Lx + 0.1*t)
        v = -0.2 * np.cos(2*np.pi*Y/Ly) * np.sin(2*np.pi*X/Lx + 0.1*t)

        # 添加速度扰动
        u += 0.05 * np.cos(4*np.pi*X/Lx - 0.08*t) * np.sin(2*np.pi*Y/Ly)
        v += 0.05 * np.sin(4*np.pi*X/Lx - 0.08*t) * np.cos(2*np.pi*Y/Ly)

        # 压力场
        p = 0.01 * np.sin(np.pi*X/Lx) * np.cos(np.pi*Y/Ly) + 0.005 * np.random.normal(0, 1, (nx, ny))

        # 存储场
        fields[t_idx, :, :, 0] = T  # 温度
        fields[t_idx, :, :, 1] = p  # 压力
        fields[t_idx, :, :, 2] = u  # u速度
        fields[t_idx, :, :, 3] = v  # v速度

    print("💾 保存数据到HDF5文件...")

    # 保存到HDF5文件
    with h5py.File(output_file, 'w') as f:
        # 保存主要数据
        f.create_dataset('fields', data=fields, compression='gzip', compression_opts=6)

        # 保存坐标
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('time', data=t_values)

        # 保存参数
        f.attrs['Ra'] = Ra
        f.attrs['Pr'] = Pr
        f.attrs['Lx'] = Lx
        f.attrs['Ly'] = Ly
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nt'] = nt
        f.attrs['dt'] = dt

    # 检查文件
    file_size = os.path.getsize(output_file) / 1024 / 1024
    print(f"✅ 数据集创建完成!")
    print(f"  文件: {output_file}")
    print(f"  大小: {file_size:.1f} MB")
    print(f"  形状: {fields.shape}")

    return output_file

def estimate_clips(data_file):
    """估算能产生多少训练clips"""
    print(f"\n📊 估算训练样本数:")

    try:
        with h5py.File(data_file, 'r') as f:
            fields_shape = f['fields'].shape
            nt = fields_shape[0]

        print(f"  时间步数: {nt}")

        # 计算clips数量
        # 每个clip需要 clip_length * temporal_downsample = 8 * 4 = 32 timesteps
        clip_length_needed = 8 * 4
        total_clips = max(0, nt - clip_length_needed + 1)

        print(f"  每个clip需要: {clip_length_needed} timesteps")
        print(f"  可产生clips: {total_clips}")

        # 按数据集分割
        train_clips = int(total_clips * 0.7)
        val_clips = int(total_clips * 0.15)
        test_clips = total_clips - train_clips - val_clips

        print(f"  数据集分割:")
        print(f"    训练集: ~{train_clips} clips")
        print(f"    验证集: ~{val_clips} clips")
        print(f"    测试集: ~{test_clips} clips")

        if train_clips > 200:
            print("✅ 训练数据充足!")
        elif train_clips > 100:
            print("⚠️  训练数据还可以")
        else:
            print("❌ 训练数据仍然不足")

        return train_clips

    except Exception as e:
        print(f"❌ 无法分析文件: {e}")
        return 0

def test_data_loading(data_file):
    """测试新数据的加载"""
    print(f"\n🧪 测试数据加载:")

    try:
        from cdanet.data import RBDataModule

        data_module = RBDataModule(
            data_dir='./rb_data_numerical',
            spatial_downsample=4,
            temporal_downsample=4,
            clip_length=8,
            batch_size=2,
            normalize=True,
            num_workers=0
        )

        data_module.setup([1e5])

        data_info = data_module.get_dataset_info()
        total_samples = sum(info['num_samples'] for info in data_info.values())

        print(f"  ✅ 数据加载成功")
        print(f"  总样本数: {total_samples}")

        for key, info in data_info.items():
            print(f"    {key}: {info['num_samples']} samples")

        if total_samples > 100:
            print("✅ 样本数量足够，可以开始训练!")
            return True
        else:
            print("⚠️  样本数量仍然偏少")
            return False

    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def main():
    print("=" * 60)
    print("大型RB数据集生成器")
    print("=" * 60)

    # 创建数据集
    data_file = create_large_rb_dataset()

    # 估算clips数量
    train_clips = estimate_clips(data_file)

    # 测试数据加载
    success = test_data_loading(data_file)

    print(f"\n" + "=" * 60)
    print("📊 总结:")
    print(f"  数据文件: {data_file}")
    print(f"  预期训练clips: {train_clips}")

    if success and train_clips > 100:
        print("🚀 准备就绪! 运行训练:")
        print("  python train_better.py")
    else:
        print("⚠️  可能需要进一步调整参数")

    print("=" * 60)

if __name__ == '__main__':
    main()