#!/usr/bin/env python3
"""
彻底修复可视化中的条纹问题
解决GT和prediction显示为条纹而不是波的问题
"""

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import os
from visualize_results import load_model_and_predict, create_demo_data
from cdanet.utils.rb_visualization import RBVisualization

def analyze_data_shapes_and_ranges(checkpoint_path, data_dir, Ra=1e5):
    """分析数据的形状和数值范围"""
    print("=" * 60)
    print("数据形状和范围分析")
    print("=" * 60)

    # 加载数据
    data_file = os.path.join(data_dir, f'rb_data_Ra_{Ra:.0e}.h5')
    if not os.path.exists(data_file):
        # 尝试找到其他文件
        import glob
        h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
        if h5_files:
            data_file = h5_files[0]
            print(f"使用文件: {data_file}")
        else:
            print("未找到数据文件")
            return None

    try:
        results = load_model_and_predict(
            checkpoint_path, data_file, Ra,
            spatial_downsample=4, temporal_downsample=4
        )

        print("\n数据形状:")
        for key, data in results.items():
            if len(data) > 0:
                print(f"  {key}: {data.shape}")
                print(f"    数值范围: [{data.min():.3f}, {data.max():.3f}]")
                print(f"    均值: {data.mean():.3f}, 标准差: {data.std():.3f}")

        return results

    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def fix_visualization_shapes(results):
    """修复可视化中的形状问题"""
    print("\n=" * 60)
    print("修复数据形状")
    print("=" * 60)

    if not results or 'truth_fields' not in results:
        print("没有可用数据")
        return None

    truth_fields = results['truth_fields']
    predictions = results['predictions']
    input_fields = results['input_fields']

    print(f"原始形状:")
    print(f"  Truth: {truth_fields.shape}")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Input: {input_fields.shape}")

    # 从rb_simulation.py中我们知道，默认网格应该是 (ny=170, nx=512) 或类似
    # 但从数据加载看应该是8个时间步
    T, total_spatial, C = truth_fields.shape

    # 根据RB仿真的标准设置，尝试几种可能的空间形状
    possible_shapes = [
        (128, 384),  # fast mode from rb_simulation
        (170, 512),  # normal mode from rb_simulation
        (64, 192),   # quarter resolution
        (85, 256),   # half resolution
    ]

    # 寻找匹配的形状
    spatial_per_time = total_spatial // T
    best_shape = None

    print(f"每个时间步的空间点数: {spatial_per_time}")

    for h, w in possible_shapes:
        if h * w == spatial_per_time:
            best_shape = (h, w)
            print(f"找到匹配形状: {h} × {w}")
            break

    if best_shape is None:
        # 尝试最接近正方形的因式分解
        factors = []
        for i in range(1, int(spatial_per_time**0.5) + 1):
            if spatial_per_time % i == 0:
                factors.append((i, spatial_per_time // i))

        # 选择最接近标准RB比例(3:1 宽高比)的
        target_ratio = 3.0
        best_diff = float('inf')
        for h, w in factors:
            ratio = w / h
            diff = abs(ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_shape = (h, w)

        print(f"使用计算得出的形状: {best_shape} (宽高比: {best_shape[1]/best_shape[0]:.2f})")

    H, W = best_shape

    # 重新整形数据为 [T, H, W, C]
    truth_reshaped = truth_fields.reshape(T, H, W, C)
    pred_reshaped = predictions.reshape(T, H, W, C)

    # 对于input_fields，它可能有不同的时间和空间分辨率
    if len(input_fields.shape) == 3:
        T_input, spatial_input, C_input = input_fields.shape
        H_input = H // 4  # 假设空间下采样4倍
        W_input = W // 4

        if H_input * W_input * T == spatial_input:
            input_reshaped = input_fields.reshape(T, H_input, W_input, C_input)
        else:
            # 尝试重新计算input的形状
            spatial_per_time_input = spatial_input // T_input
            h_input = int((spatial_per_time_input / (W/H))**0.5)
            w_input = spatial_per_time_input // h_input
            input_reshaped = input_fields.reshape(T_input, h_input, w_input, C_input)
    else:
        input_reshaped = input_fields

    print(f"重新整形后:")
    print(f"  Truth: {truth_reshaped.shape}")
    print(f"  Predictions: {pred_reshaped.shape}")
    print(f"  Input: {input_reshaped.shape}")

    return {
        'truth_fields': truth_reshaped,
        'predictions': pred_reshaped,
        'input_fields': input_reshaped
    }

def create_fixed_visualization(data, output_dir="./fixed_visualizations"):
    """创建修复后的可视化"""
    print("\n=" * 60)
    print("创建修复后的可视化")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    truth_fields = data['truth_fields']  # [T, H, W, C]
    predictions = data['predictions']
    input_fields = data['input_fields']

    # 提取温度场 (C=0)
    truth_temp = truth_fields[:, :, :, 0]    # [T, H, W]
    pred_temp = predictions[:, :, :, 0]      # [T, H, W]

    # 计算实际的数值范围
    all_temp = np.concatenate([truth_temp.flatten(), pred_temp.flatten()])
    vmin, vmax = np.percentile(all_temp, [5, 95])  # 使用5-95百分位数作为范围

    print(f"温度场数值范围: [{vmin:.3f}, {vmax:.3f}]")
    print(f"Truth 温度范围: [{truth_temp.min():.3f}, {truth_temp.max():.3f}]")
    print(f"Prediction 温度范围: [{pred_temp.min():.3f}, {pred_temp.max():.3f}]")

    # 创建自定义可视化
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # 选择几个时间点进行可视化
    time_indices = [0, truth_temp.shape[0]//3, 2*truth_temp.shape[0]//3, truth_temp.shape[0]-1]

    for i, t_idx in enumerate(time_indices):
        if t_idx >= truth_temp.shape[0]:
            t_idx = truth_temp.shape[0] - 1

        # Ground Truth
        im1 = axes[0, i].imshow(truth_temp[t_idx],
                               cmap='hot', vmin=vmin, vmax=vmax,
                               origin='lower', aspect='auto',
                               extent=[0, 3, 0, 1])
        axes[0, i].set_title(f'Truth t={t_idx}')
        axes[0, i].set_xlabel('x')
        if i == 0:
            axes[0, i].set_ylabel('y')

        # Prediction
        im2 = axes[1, i].imshow(pred_temp[t_idx],
                               cmap='hot', vmin=vmin, vmax=vmax,
                               origin='lower', aspect='auto',
                               extent=[0, 3, 0, 1])
        axes[1, i].set_title(f'Prediction t={t_idx}')
        axes[1, i].set_xlabel('x')
        if i == 0:
            axes[1, i].set_ylabel('y')

        # Error
        error = pred_temp[t_idx] - truth_temp[t_idx]
        error_max = max(np.abs(error).max(), 0.01)  # 避免除零
        im3 = axes[2, i].imshow(error,
                               cmap='RdBu_r', vmin=-error_max, vmax=error_max,
                               origin='lower', aspect='auto',
                               extent=[0, 3, 0, 1])
        axes[2, i].set_title(f'Error t={t_idx}')
        axes[2, i].set_xlabel('x')
        if i == 0:
            axes[2, i].set_ylabel('y')

        # 添加colorbar
        if i == 3:  # 最后一列添加colorbar
            plt.colorbar(im1, ax=axes[0, i], label='Temperature')
            plt.colorbar(im2, ax=axes[1, i], label='Temperature')
            plt.colorbar(im3, ax=axes[2, i], label='Error')

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, 'fixed_temperature_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存可视化图像: {output_path}")

    # 创建单个时间步的详细分析
    plt.figure(figsize=(12, 4))

    mid_time = truth_temp.shape[0] // 2

    plt.subplot(1, 3, 1)
    plt.imshow(truth_temp[mid_time], cmap='hot', vmin=vmin, vmax=vmax,
               origin='lower', aspect='auto', extent=[0, 3, 0, 1])
    plt.title('Ground Truth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Temperature')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_temp[mid_time], cmap='hot', vmin=vmin, vmax=vmax,
               origin='lower', aspect='auto', extent=[0, 3, 0, 1])
    plt.title('CDAnet Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Temperature')

    plt.subplot(1, 3, 3)
    error = pred_temp[mid_time] - truth_temp[mid_time]
    error_max = np.abs(error).max()
    plt.imshow(error, cmap='RdBu_r', vmin=-error_max, vmax=error_max,
               origin='lower', aspect='auto', extent=[0, 3, 0, 1])
    plt.title('Prediction Error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Error')

    plt.tight_layout()

    detail_path = os.path.join(output_dir, 'detailed_comparison.png')
    plt.savefig(detail_path, dpi=300, bbox_inches='tight')
    print(f"保存详细对比图像: {detail_path}")

    plt.close('all')

    return output_path, detail_path

def check_input_data_quality(data_file):
    """检查原始输入数据的质量"""
    print("\n=" * 60)
    print("检查原始数据质量")
    print("=" * 60)

    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return

    with h5py.File(data_file, 'r') as f:
        print(f"文件: {data_file}")
        print(f"数据集: {list(f.keys())}")
        print(f"属性: {dict(f.attrs)}")

        if 'data' in f:
            data = f['data'][:]
            print(f"数据形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")

            # 检查每个变量的统计
            var_names = ['Temperature', 'Pressure', 'U-velocity', 'V-velocity']
            for i, name in enumerate(var_names):
                if i < data.shape[-1]:
                    var_data = data[:, :, :, i]
                    print(f"{name}:")
                    print(f"  范围: [{var_data.min():.3f}, {var_data.max():.3f}]")
                    print(f"  均值: {var_data.mean():.3f}")
                    print(f"  标准差: {var_data.std():.3f}")

        # 可视化原始数据的几个时间步
        if 'data' in f:
            raw_data = f['data'][:]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            time_steps = [0, raw_data.shape[0]//2, raw_data.shape[0]-1]

            for i, t in enumerate(time_steps):
                # 温度场
                temp = raw_data[t, :, :, 0]
                im = axes[0, i].imshow(temp, cmap='hot', origin='lower', aspect='auto')
                axes[0, i].set_title(f'Raw Temperature t={t}')
                plt.colorbar(im, ax=axes[0, i])

                # U速度场
                u_vel = raw_data[t, :, :, 2]
                im = axes[1, i].imshow(u_vel, cmap='RdBu_r', origin='lower', aspect='auto')
                axes[1, i].set_title(f'Raw U-velocity t={t}')
                plt.colorbar(im, ax=axes[1, i])

            plt.tight_layout()
            plt.savefig('./fixed_visualizations/raw_data_check.png', dpi=300, bbox_inches='tight')
            print("保存原始数据检查图像: ./fixed_visualizations/raw_data_check.png")
            plt.close()

def main():
    """主函数"""
    # 参数设置
    checkpoint_path = "checkpoints/best_model.pth"  # 请调整为实际路径
    data_dir = "./rb_data_numerical"  # 请调整为实际路径
    Ra = 1e5

    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"警告: 模型文件不存在 {checkpoint_path}")
        print("请提供正确的checkpoint路径")
        return

    # 首先检查原始数据
    data_file = os.path.join(data_dir, f'rb_data_Ra_{Ra:.0e}.h5')
    if not os.path.exists(data_file):
        import glob
        h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
        if h5_files:
            data_file = h5_files[0]

    if os.path.exists(data_file):
        check_input_data_quality(data_file)

    # 分析数据
    results = analyze_data_shapes_and_ranges(checkpoint_path, data_dir, Ra)

    if results is not None:
        # 修复形状
        fixed_data = fix_visualization_shapes(results)

        if fixed_data is not None:
            # 创建修复后的可视化
            create_fixed_visualization(fixed_data)
            print("\n✅ 修复完成！检查 ./fixed_visualizations/ 目录中的结果")
        else:
            print("❌ 数据形状修复失败")
    else:
        print("❌ 数据加载失败，使用演示数据")
        demo_data = create_demo_data()
        create_fixed_visualization(demo_data)

if __name__ == "__main__":
    main()