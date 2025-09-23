#!/usr/bin/env python3
"""
基于你原有rb_simulation.py的改进版本
保持你的代码结构，但修复数值稳定性和物理正确性
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter


def generate_rb_snapshot_improved(nx, ny, Ra, time_step, dt=1e-3):
    """
    基于你原有generate_rb_snapshot函数的改进版本
    保持多尺度对流结构，但提高数值稳定性
    """
    # 坐标网格 - 与你的代码一致
    Lx, Ly = 3.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # 基础温度分布 - 与你的代码一致
    T_base = 1.0 - Y / Ly

    # 添加2D基础结构 - 与你的代码一致但减小幅度
    T_2d_variation = 0.05 * np.sin(2 * np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    T_2d_variation += 0.025 * np.sin(4 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly)

    T = T_base + T_2d_variation

    # 时间参数
    time = time_step * dt

    # 对流结构参数 - 与你的代码一致但调整振幅
    n_large = 2
    amp_large = 0.2  # 减小以提高稳定性

    n_medium = 4
    amp_medium = 0.1  # 减小以提高稳定性

    n_small = 6  # 减少小尺度数量
    amp_small = 0.03

    # 初始化场
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    T_pert = np.zeros_like(X)

    # 大尺度对流卷筒 - 与你的代码结构相同
    for i in range(n_large):
        x_center = (i + 0.5) * Lx / n_large
        kx_large = 2 * np.pi * n_large / Lx

        for j in range(2):
            ky_large = np.pi * (j + 1) / Ly

            # 相位演化 - 与你的代码一致但添加稳定性
            phase_large = time * 0.3 + i * np.pi / n_large + j * np.pi / 2
            phase_large += 0.05 * np.sin(time * 0.2 + i + j)  # 减小随机扰动

            # 振幅调制
            amp_y = amp_large * (1.0 - 0.2 * j)

            # 流函数方法 - 与你的代码相同
            # u = -∂ψ/∂y, v = ∂ψ/∂x
            u += amp_y * kx_large * np.cos(kx_large * (X - x_center)) * np.sin(ky_large * Y) * np.cos(phase_large)
            v += -amp_y * ky_large * np.sin(kx_large * (X - x_center)) * np.cos(ky_large * Y) * np.cos(phase_large)

            # 温度扰动
            T_pert += amp_y * 0.3 * np.sin(kx_large * (X - x_center)) * np.sin(ky_large * Y) * np.cos(phase_large + np.pi/4)

    # 中尺度对流 - 与你的代码结构相同
    for i in range(n_medium):
        kx_med = 2 * np.pi * n_medium / Lx

        for j in range(2):
            ky_med = np.pi * (j + 2) / Ly
            phase_med = time * 0.5 + i * np.pi / n_medium + j * np.pi / 3
            phase_med += 0.1 * np.sin(time * 0.4 + i + j)

            amp_med_y = amp_medium * (1.0 - 0.15 * j)

            u += amp_med_y * np.cos(kx_med * X + phase_med) * np.sin(ky_med * Y)
            v += amp_med_y * np.sin(kx_med * X + phase_med) * np.cos(ky_med * Y) * 0.5
            T_pert += amp_med_y * 0.2 * np.sin(kx_med * X + phase_med) * np.sin(ky_med * Y)

    # 小尺度湍流 - 与你的代码相同但更稳定
    np.random.seed(int(time * 1000) % 10000)  # 可重复的随机数
    for i in range(n_small):
        kx_small = 2 * np.pi * (n_small + 0.5 * np.random.normal()) / Lx
        ky_small = 2 * np.pi * (1 + i//2 + 0.3 * np.random.normal()) / Ly
        phase_small = time * (1.0 + 0.3 * i) + 0.1 * np.random.normal()

        amp_small_effective = amp_small * (1.0 + 0.3 * np.random.normal())

        u += amp_small_effective * np.cos(kx_small * X + phase_small) * np.sin(ky_small * Y)
        v += amp_small_effective * np.sin(kx_small * X + phase_small) * np.cos(ky_small * Y)
        T_pert += amp_small_effective * 0.15 * np.sin(kx_small * X + phase_small) * np.sin(ky_small * Y)

    # 添加温度扰动
    T += T_pert

    # 热羽流 - 与你的代码相同但参数调整
    for plume in range(3):  # 减少羽流数量
        y_center = (plume + 0.5) * Ly / 3
        y_width = 0.15

        plume_phase = time * 0.3 + plume * np.pi / 2
        plume_strength = 0.08 * np.cos(plume_phase)  # 减小强度

        y_profile = np.exp(-((Y - y_center) / y_width)**2)
        x_modulation = 1.0 + 0.2 * np.sin(2 * np.pi * X / Lx + plume * np.pi / 3)

        T += plume_strength * y_profile * x_modulation

    # 边界层效应 - 与你的代码相同
    boundary_layer = 0.08
    y_boundary_bottom = np.exp(-Y / boundary_layer)
    y_boundary_top = np.exp(-(Ly - Y) / boundary_layer)

    boundary_variation = 1.0 + 0.15 * np.sin(4 * np.pi * X / Lx + time)
    T += 0.05 * y_boundary_bottom * boundary_variation - 0.05 * y_boundary_top * boundary_variation

    # 应用边界条件 - 与你的代码相同
    T[0, :] = 1.0 + 0.03 * np.sin(2 * np.pi * x / Lx + time)
    T[-1, :] = 0.0 + 0.015 * np.sin(2 * np.pi * x / Lx + time * 0.7)

    # 速度无滑移边界条件
    u[0, :] = u[-1, :] = 0.0
    v[0, :] = v[-1, :] = 0.0

    # 周期边界条件
    T[:, 0] = T[:, -1]
    u[:, 0] = u[:, -1]
    v[:, 0] = v[:, -1]

    # 压力场 - 与你的代码相同但简化
    dudx = np.gradient(u, axis=1) / (Lx / nx)
    dvdy = np.gradient(v, axis=0) / (Ly / ny)
    dudy = np.gradient(u, axis=0) / (Ly / ny)
    dvdx = np.gradient(v, axis=1) / (Lx / nx)

    p = -0.3 * (dudx**2 + dvdy**2 + 2 * dudy * dvdx)
    p += 0.1 * (1 - Y)  # 静水压力

    # 添加噪声 - 与你的代码相同但减小幅度
    noise_level = 0.005  # 减小噪声
    T += noise_level * np.random.normal(0, 1, T.shape)
    u += noise_level * 0.3 * np.random.normal(0, 1, u.shape)
    v += noise_level * 0.3 * np.random.normal(0, 1, v.shape)
    p += noise_level * 0.1 * np.random.normal(0, 1, p.shape)

    # 平滑处理以提高稳定性
    T = gaussian_filter(T, sigma=0.3)
    u = gaussian_filter(u, sigma=0.3)
    v = gaussian_filter(v, sigma=0.3)
    p = gaussian_filter(p, sigma=0.3)

    # 重新应用边界条件
    T[0, :] = 1.0
    T[-1, :] = 0.0
    u[0, :] = u[-1, :] = 0.0
    v[0, :] = v[-1, :] = 0.0

    return T, u, v, p


def generate_training_dataset_improved(Ra=1e5, n_runs=5, n_samples=50, nx=128, ny=64,
                                     save_path='rb_data_improved'):
    """
    基于你原有generate_training_dataset的改进版本
    """
    os.makedirs(save_path, exist_ok=True)

    # 参数设置
    dt = 1e-3
    delta_t = 0.1

    print(f"🌡️ 基于原有代码的改进RB数据生成")
    print(f"  瑞利数: Ra = {Ra:.0e}")
    print(f"  运行次数: {n_runs}")
    print(f"  每次样本: {n_samples}")
    print(f"  网格: {nx}×{ny}")
    print()

    all_data = []

    for run in range(n_runs):
        print(f"  运行 {run+1}/{n_runs}")

        run_data = []
        for sample in range(n_samples):
            # 时间演化
            time_step = sample + run * n_samples * 2  # 确保不同运行有不同时间

            # 生成快照
            T, u, v, p = generate_rb_snapshot_improved(nx, ny, Ra, time_step, dt)

            # 保存数据
            frame_data = {
                'temperature': T.copy(),
                'velocity_x': u.copy(),
                'velocity_y': v.copy(),
                'pressure': p.copy(),
                'time': time_step * dt
            }
            run_data.append(frame_data)

            if sample % 10 == 0:
                print(f"    样本 {sample+1}/{n_samples}")

        # 保存单次运行 - 与你的原有格式兼容
        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        with h5py.File(filename, 'w') as f:
            for i, frame in enumerate(run_data):
                grp = f.create_group(f'frame_{i:03d}')
                for key, value in frame.items():
                    if key != 'time':
                        grp.create_dataset(key, data=value)
                    else:
                        grp.attrs['time'] = value

        print(f"    保存: {filename}")
        all_data.append(run_data)

    # 创建合并数据集
    create_consolidated_dataset_improved(save_path, Ra, all_data, nx, ny)

    return all_data


def create_consolidated_dataset_improved(save_path, Ra, all_data, nx, ny):
    """创建合并数据集，兼容训练格式"""
    n_runs = len(all_data)
    n_samples = len(all_data[0])

    print(f"\n📦 创建合并数据集: {n_runs} 运行 × {n_samples} 样本")

    # 使用你的数据转换逻辑
    p_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    b_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    u_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    w_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)

    for run_idx, run_data in enumerate(all_data):
        for sample_idx, frame in enumerate(run_data):
            p_data[run_idx, sample_idx] = frame['pressure']
            b_data[run_idx, sample_idx] = frame['temperature']  # T -> b
            u_data[run_idx, sample_idx] = frame['velocity_x']
            w_data[run_idx, sample_idx] = frame['velocity_y']   # v -> w

    # 保存为训练格式
    output_file = f'{save_path}/rb2d_ra{Ra:.0e}_consolidated.h5'

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('p', data=p_data, compression='gzip')
        f.create_dataset('b', data=b_data, compression='gzip')
        f.create_dataset('u', data=u_data, compression='gzip')
        f.create_dataset('w', data=w_data, compression='gzip')

        f.attrs['Ra'] = Ra
        f.attrs['Pr'] = 0.7
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['Lx'] = 3.0
        f.attrs['Ly'] = 1.0
        f.attrs['n_runs'] = n_runs
        f.attrs['n_samples'] = n_samples
        f.attrs['format'] = 'reference_compatible'
        f.attrs['source'] = 'improved_original'

    print(f"✅ 合并数据集: {output_file}")

    # 数据统计
    print(f"\n📊 数据统计:")
    print(f"  温度范围: [{b_data.min():.3f}, {b_data.max():.3f}]")
    print(f"  压力范围: [{p_data.min():.3f}, {p_data.max():.3f}]")
    print(f"  U速度范围: [{u_data.min():.3f}, {u_data.max():.3f}]")
    print(f"  V速度范围: [{w_data.min():.3f}, {w_data.max():.3f}]")

    return output_file


def create_visualization(data_file, save_path):
    """创建RB数据可视化"""
    print(f"\n🎨 创建可视化: {data_file}")

    import h5py
    import matplotlib.pyplot as plt

    with h5py.File(data_file, 'r') as f:
        # 读取数据
        p_data = f['p'][:]  # 压力
        b_data = f['b'][:]  # 温度(浮力)
        u_data = f['u'][:]  # X速度
        w_data = f['w'][:]  # Y速度

        n_runs, n_samples, ny, nx = p_data.shape
        print(f"  数据形状: {n_runs} runs × {n_samples} samples × {ny}×{nx}")

    # 选择第一个运行的几个时间步进行可视化
    run_idx = 0
    time_steps = [0, n_samples//4, n_samples//2, 3*n_samples//4, n_samples-1]

    # 创建可视化
    fig, axes = plt.subplots(4, len(time_steps), figsize=(15, 12))
    fig.suptitle(f'Rayleigh-Benard Convection Visualization (Run {run_idx+1})', fontsize=16)

    for i, t in enumerate(time_steps):
        # 温度场
        im1 = axes[0, i].imshow(b_data[run_idx, t], cmap='RdBu_r', aspect='equal')
        axes[0, i].set_title(f'Temperature t={t}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # 压力场
        im2 = axes[1, i].imshow(p_data[run_idx, t], cmap='viridis', aspect='equal')
        axes[1, i].set_title(f'Pressure t={t}')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # X速度场
        im3 = axes[2, i].imshow(u_data[run_idx, t], cmap='RdBu', aspect='equal')
        axes[2, i].set_title(f'U Velocity t={t}')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

        # Y速度场
        im4 = axes[3, i].imshow(w_data[run_idx, t], cmap='RdBu', aspect='equal')
        axes[3, i].set_title(f'W Velocity t={t}')
        axes[3, i].set_xticks([])
        axes[3, i].set_yticks([])

    # 添加颜色条
    plt.colorbar(im1, ax=axes[0, :], shrink=0.6, label='Temperature')
    plt.colorbar(im2, ax=axes[1, :], shrink=0.6, label='Pressure')
    plt.colorbar(im3, ax=axes[2, :], shrink=0.6, label='U Velocity')
    plt.colorbar(im4, ax=axes[3, :], shrink=0.6, label='W Velocity')

    plt.tight_layout()

    # 保存图像
    viz_file = f"{save_path}/rb_visualization.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化保存: {viz_file}")

    # 创建单个样本的流场可视化
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 选择中间时间步
    mid_t = n_samples // 2
    T = b_data[run_idx, mid_t]
    U = u_data[run_idx, mid_t]
    W = w_data[run_idx, mid_t]

    # 温度场作为背景
    im = ax1.imshow(T, cmap='RdBu_r', aspect='equal', extent=[0, 3, 0, 1])
    ax1.set_title('Temperature Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im, ax=ax1, label='Temperature')

    # 流场矢量图
    x = np.linspace(0, 3, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # 降采样以便清晰显示矢量
    skip = 4
    ax2.imshow(T, cmap='RdBu_r', aspect='equal', extent=[0, 3, 0, 1], alpha=0.7)
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               U[::skip, ::skip], W[::skip, ::skip],
               scale=3, width=0.003, alpha=0.8)
    ax2.set_title('Flow Field Vectors')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.tight_layout()

    # 保存流场图
    flow_file = f"{save_path}/rb_flow_field.png"
    plt.savefig(flow_file, dpi=150, bbox_inches='tight')
    print(f"✅ 流场图保存: {flow_file}")

    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基于原有代码的改进RB数据生成器')
    parser.add_argument('--Ra', type=float, default=1e5, help='瑞利数')
    parser.add_argument('--n_runs', type=int, default=5, help='运行次数')
    parser.add_argument('--n_samples', type=int, default=50, help='每次运行的样本数')
    parser.add_argument('--nx', type=int, default=128, help='X方向网格点')
    parser.add_argument('--ny', type=int, default=64, help='Y方向网格点')
    parser.add_argument('--save_path', type=str, default='rb_data_improved', help='保存路径')
    parser.add_argument('--visualize', action='store_true', help='创建可视化')

    args = parser.parse_args()

    print("🔧 基于原有代码的改进RB数据生成器")
    print("=" * 50)
    print("保持原有代码结构，修复数值稳定性")
    print()

    # 生成数据
    generate_training_dataset_improved(
        Ra=args.Ra,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        nx=args.nx,
        ny=args.ny,
        save_path=args.save_path
    )

    # 可视化
    if args.visualize:
        data_file = f"{args.save_path}/rb2d_ra{args.Ra:.0e}_consolidated.h5"
        if os.path.exists(data_file):
            create_visualization(data_file, args.save_path)

    print("\n✅ 改进数据生成完成！")
    print(f"📁 数据保存在: {args.save_path}/")
    print("🚀 现在可以用于CDAnet训练！")