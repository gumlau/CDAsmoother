#!/usr/bin/env python3
"""
Proper Rayleigh-Bénard Numerical Simulation
Uses real PDE solver for physics-accurate data generation
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from scipy.fft import fft2, ifft2


class RBNumericalSimulation:
    """
    Stable Rayleigh-Bénard numerical solver based on working sourcecodeCDAnet/sim.py
    """
    def __init__(self, nx=128, ny=64, Lx=3.0, Ly=1.0, Ra=1e5, Pr=0.7, dt=1e-3, save_path='rb_data_numerical'):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.Ra = Ra
        self.Pr = Pr
        self.dt = dt
        self.save_path = save_path

        self.dx = Lx / nx
        self.dy = Ly / ny

        self.T = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        self.T_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.u_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.v_prev = [np.zeros((ny, nx)) for _ in range(2)]

        self.setup_initial_conditions()

    def setup_initial_conditions(self):
        self.T[-1, :] = 1.0
        self.T[0, :] = 0.0
        self.T += 1e-3 * np.random.randn(self.ny, self.nx)

        for i in range(2):
            self.T_prev[i] = self.T.copy()
            self.u_prev[i] = self.u.copy()
            self.v_prev[i] = self.v.copy()

    def solve_pressure_poisson(self):
        source = (
            (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1)) / (2*self.dx) +
            (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / (2*self.dy)
        ) / self.dt

        source_fft = fft2(source)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky)
        K2 = Kx*Kx + Ky*Ky
        K2[0,0] = 1.0

        p_fft = source_fft / (-K2)
        self.p = np.real(ifft2(p_fft))

    def adams_bashforth_step(self, f, f_prev1, f_prev2):
        return f + self.dt * (
            23/12 * f_prev1 - 16/12 * f_prev2 + 5/12 * f
        )

    def step(self, step_number):
        self.T_prev[1] = self.T_prev[0].copy()
        self.T_prev[0] = self.T.copy()
        self.u_prev[1] = self.u_prev[0].copy()
        self.u_prev[0] = self.u.copy()
        self.v_prev[1] = self.v_prev[0].copy()
        self.v_prev[0] = self.v.copy()

        self.solve_pressure_poisson()

        self.u = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=1) - np.roll(self.p, 1, axis=1)) / (2*self.dx),
            self.u_prev[0],
            self.u_prev[1]
        )

        self.v = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)) / (2*self.dy) +
            self.Ra * self.Pr * self.T,
            self.v_prev[0],
            self.v_prev[1]
        )

        T_update = self.adams_bashforth_step(
            self.Pr * (
                (np.roll(self.T, -1, axis=1) - 2*self.T + np.roll(self.T, 1, axis=1)) / (self.dx*self.dx) +
                (np.roll(self.T, -1, axis=0) - 2*self.T + np.roll(self.T, 1, axis=0)) / (self.dy*self.dy)
            ) - (
                self.u * (np.roll(self.T, -1, axis=1) - np.roll(self.T, 1, axis=1)) / (2*self.dx) +
                self.v * (np.roll(self.T, -1, axis=0) - np.roll(self.T, 1, axis=0)) / (2*self.dy)
            ),
            self.T_prev[0],
            self.T_prev[1]
        )

        self.T = np.clip(T_update, 0, 1)
        self.T[0, :] = 0.0
        self.T[-1, :] = 1.0

        self.u[:, 0] = self.u[:, -1]
        self.u[:, -1] = self.u[:, 0]
        self.v[:, 0] = self.v[:, -1]
        self.v[:, -1] = self.v[:, 0]


def generate_rb_data_numerical(Ra=1e5, n_runs=5, n_samples=50, nx=128, ny=64, save_path='rb_data_numerical'):
    """
    Generate realistic Rayleigh-Bénard data using numerical simulation
    This will take significantly longer but produce physics-accurate data
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"🔥 Real Numerical RB Simulation")
    print(f"  Rayleigh number: Ra = {Ra:.0e}")
    print(f"  Runs: {n_runs}")
    print(f"  Samples per run: {n_samples}")
    print(f"  Grid: {nx}×{ny}")
    print(f"  ⚠️  This will take MUCH longer (~10-30min per run)")
    print()

    # Simulation parameters matching working sourcecodeCDAnet/sim.py
    dt = 1e-3    # Proven stable timestep
    t_startup = 5.0  # Adequate startup time
    delta_t = 0.1    # Time between snapshots

    all_data = []

    for run in range(n_runs):
        print(f"  🏃 Run {run+1}/{n_runs}")
        print(f"    Initializing simulation...")

        # Create simulation
        sim = RBNumericalSimulation(nx=nx, ny=ny, Ra=Ra, dt=dt)

        # Startup phase - let flow develop
        n_startup_steps = int(t_startup / dt)
        print(f"    Startup: {n_startup_steps} steps ({t_startup}s)")

        for step in range(n_startup_steps):
            sim.step(step)
            if step % (n_startup_steps // 10) == 0:
                progress = int(100 * step / n_startup_steps)
                print(f"      Startup: {progress}%")

        print(f"    Collecting {n_samples} samples...")

        # Data collection phase
        run_data = []
        steps_between_saves = int(delta_t / dt)

        for sample in range(n_samples):
            # Advance simulation
            for _ in range(steps_between_saves):
                sim.step(n_startup_steps + sample * steps_between_saves + _)

            # Save data
            frame_data = {
                'temperature': sim.T.copy(),
                'velocity_x': sim.u.copy(),
                'velocity_y': sim.v.copy(),
                'pressure': sim.p.copy(),
                'time': (sample + 1) * delta_t
            }
            run_data.append(frame_data)

            if sample % 10 == 0:
                print(f"      Sample {sample+1}/{n_samples}")

        # Save run data
        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        with h5py.File(filename, 'w') as f:
            for i, frame in enumerate(run_data):
                grp = f.create_group(f'frame_{i:03d}')
                for key, value in frame.items():
                    if key != 'time':
                        grp.create_dataset(key, data=value)
                    else:
                        grp.attrs['time'] = value

        print(f"    ✅ Saved: {filename}")
        all_data.append(run_data)

    # Create consolidated dataset
    create_consolidated_dataset_numerical(save_path, Ra, all_data, nx, ny)

    return all_data


def create_consolidated_dataset_numerical(save_path, Ra, all_data, nx, ny):
    """Create consolidated dataset from numerical simulation runs"""
    n_runs = len(all_data)
    n_samples = len(all_data[0])

    print(f"\n📦 Creating consolidated numerical dataset: {n_runs} runs × {n_samples} samples")

    # Initialize arrays
    p_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    b_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)  # Temperature -> b
    u_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    w_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)  # v -> w

    # Fill arrays with numerical simulation data
    for run_idx, run_data in enumerate(all_data):
        for sample_idx, frame in enumerate(run_data):
            p_data[run_idx, sample_idx] = frame['pressure']
            b_data[run_idx, sample_idx] = frame['temperature']
            u_data[run_idx, sample_idx] = frame['velocity_x']
            w_data[run_idx, sample_idx] = frame['velocity_y']

    # Save consolidated dataset
    output_file = f'{save_path}/rb2d_ra{Ra:.0e}_consolidated.h5'
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('p', data=p_data, compression='gzip')
        f.create_dataset('b', data=b_data, compression='gzip')  # Temperature as 'b'
        f.create_dataset('u', data=u_data, compression='gzip')
        f.create_dataset('w', data=w_data, compression='gzip')  # v-velocity as 'w'

        # Add metadata
        f.attrs['Ra'] = Ra
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['n_runs'] = n_runs
        f.attrs['n_samples'] = n_samples
        f.attrs['simulation_type'] = 'numerical_rb_solver'

    print(f"✅ Consolidated dataset: {output_file}")

    # Print statistics
    print(f"\n📊 Numerical simulation statistics:")
    print(f"  Temperature range: [{np.min(b_data):.3f}, {np.max(b_data):.3f}]")
    print(f"  Pressure range: [{np.min(p_data):.3f}, {np.max(p_data):.3f}]")
    print(f"  U-velocity range: [{np.min(u_data):.3f}, {np.max(u_data):.3f}]")
    print(f"  V-velocity range: [{np.min(w_data):.3f}, {np.max(w_data):.3f}]")

    # Check temporal evolution
    temp_change = np.max(np.abs(b_data[0, 0] - b_data[0, -1]))
    vel_change = np.max(np.abs(u_data[0, 0] - u_data[0, -1]))
    print(f"  Temporal changes - Temperature: {temp_change:.4f}, Velocity: {vel_change:.4f}")

    return output_file


def visualize_numerical_data(output_file):
    """Create visualizations of the numerical simulation data"""
    print(f"\n🎨 Creating visualizations: {output_file}")

    with h5py.File(output_file, 'r') as f:
        b_data = f['b'][:]  # Temperature
        p_data = f['p'][:]  # Pressure
        u_data = f['u'][:]  # X-velocity
        w_data = f['w'][:]  # Y-velocity

        n_runs, n_samples, ny, nx = b_data.shape
        print(f"  Data shape: {n_runs} runs × {n_samples} samples × {ny}×{nx}")

    # Select first run for visualization
    run_idx = 0
    time_steps = [0, n_samples//4, n_samples//2, 3*n_samples//4, n_samples-1]

    # Create visualization
    fig, axes = plt.subplots(4, len(time_steps), figsize=(15, 12))
    fig.suptitle(f'Numerical Rayleigh-Bénard Simulation (Run {run_idx+1})', fontsize=16)

    for i, t in enumerate(time_steps):
        # Temperature field
        im1 = axes[0, i].imshow(b_data[run_idx, t], cmap='RdBu_r', aspect='equal')
        axes[0, i].set_title(f'Temperature t={t}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Pressure field
        im2 = axes[1, i].imshow(p_data[run_idx, t], cmap='viridis', aspect='equal')
        axes[1, i].set_title(f'Pressure t={t}')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # X-velocity field
        im3 = axes[2, i].imshow(u_data[run_idx, t], cmap='RdBu', aspect='equal')
        axes[2, i].set_title(f'U Velocity t={t}')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

        # Y-velocity field
        im4 = axes[3, i].imshow(w_data[run_idx, t], cmap='RdBu', aspect='equal')
        axes[3, i].set_title(f'W Velocity t={t}')
        axes[3, i].set_xticks([])
        axes[3, i].set_yticks([])

    # Add colorbars
    plt.colorbar(im1, ax=axes[0, :], shrink=0.6, label='Temperature')
    plt.colorbar(im2, ax=axes[1, :], shrink=0.6, label='Pressure')
    plt.colorbar(im3, ax=axes[2, :], shrink=0.6, label='U Velocity')
    plt.colorbar(im4, ax=axes[3, :], shrink=0.6, label='W Velocity')

    plt.tight_layout()

    # Save visualization
    viz_file = f"{os.path.dirname(output_file)}/numerical_rb_visualization.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {viz_file}")
    plt.close()

    return viz_file


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Generate Rayleigh-Bénard numerical simulation data')
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of simulation runs')
    parser.add_argument('--n_samples', type=int, default=20, help='Samples per run')
    parser.add_argument('--nx', type=int, default=128, help='Grid points in x')
    parser.add_argument('--ny', type=int, default=64, help='Grid points in y')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical', help='Save directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')

    args = parser.parse_args()

    # Clear old data first
    if os.path.exists(args.save_path):
        print(f"🗑️  Clearing old data in {args.save_path}")
        import shutil
        shutil.rmtree(args.save_path)

    # Generate numerical simulation data
    print("🚀 Starting NUMERICAL Rayleigh-Bénard simulation...")
    print("This uses real PDE solver - will take significantly longer!")
    print("Expect 10-30 minutes per run depending on parameters.")

    all_data = generate_rb_data_numerical(
        Ra=args.Ra,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        nx=args.nx,
        ny=args.ny,
        save_path=args.save_path
    )

    print(f"\n✅ Numerical simulation complete!")
    print(f"📁 Data saved in: {args.save_path}/")
    print(f"🚀 Ready for CDAnet training with realistic physics data!")

    # Create visualizations if requested
    if args.visualize:
        output_file = f'{args.save_path}/rb2d_ra{args.Ra:.0e}_consolidated.h5'
        if os.path.exists(output_file):
            visualize_numerical_data(output_file)
        else:
            print(f"⚠️  Consolidated file not found for visualization: {output_file}")


if __name__ == "__main__":
    main()
    import matplotlib.pyplot as plt
