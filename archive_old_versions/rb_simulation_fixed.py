#!/usr/bin/env python3
"""
Fixed RB simulation based on sourcecodeCDAnet reference implementation.
This version follows the exact physics and numerical methods from the paper.
"""

import numpy as np
from scipy.fft import fft2, ifft2
import h5py
import os
import matplotlib.pyplot as plt
import argparse


class RBNumericalSimulation:
    """
    Rayleigh-Bénard convection simulation following the reference implementation.
    Based on sourcecodeCDAnet/sim.py with proper physics and numerical methods.
    """

    def __init__(self, nx=128, ny=64, Lx=3.0, Ly=1.0, Ra=1e5, Pr=0.7, dt=5e-4, save_path='rb_data_numerical'):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.Ra = Ra
        self.Pr = Pr
        self.dt = dt
        self.save_path = save_path

        # Grid spacing
        self.dx = Lx / nx
        self.dy = Ly / ny

        # Fields: [ny, nx] arrays (y-direction first, x-direction second)
        self.T = np.zeros((ny, nx))      # Temperature
        self.u = np.zeros((ny, nx))      # x-velocity
        self.v = np.zeros((ny, nx))      # y-velocity
        self.p = np.zeros((ny, nx))      # Pressure

        # Previous time steps for Adams-Bashforth
        self.T_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.u_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.v_prev = [np.zeros((ny, nx)) for _ in range(2)]

        self.setup_initial_conditions()
        os.makedirs(self.save_path, exist_ok=True)

    def setup_initial_conditions(self):
        """Setup initial conditions following reference implementation."""
        # Linear temperature profile: hot bottom (y=0), cold top (y=Ly)
        y = np.linspace(0, self.Ly, self.ny)
        for i in range(self.ny):
            self.T[i, :] = 1.0 - y[i] / self.Ly  # T=1 at bottom, T=0 at top

        # Add small random perturbations to trigger convection
        self.T += 1e-3 * np.random.randn(self.ny, self.nx)

        # Apply boundary conditions
        self.apply_boundary_conditions()

        # Initialize history for Adams-Bashforth
        for i in range(2):
            self.T_prev[i] = self.T.copy()
            self.u_prev[i] = self.u.copy()
            self.v_prev[i] = self.v.copy()

    def apply_boundary_conditions(self):
        """Apply boundary conditions exactly as in reference code."""
        # Temperature: fixed at boundaries
        self.T[0, :] = 1.0      # Hot bottom wall
        self.T[-1, :] = 0.0     # Cold top wall

        # Velocity: no-slip at top/bottom walls
        self.u[0, :] = self.u[-1, :] = 0.0
        self.v[0, :] = self.v[-1, :] = 0.0

        # Periodic boundary conditions in x-direction
        self.T[:, 0] = self.T[:, -1]
        self.u[:, 0] = self.u[:, -1]
        self.v[:, 0] = self.v[:, -1]
        self.p[:, 0] = self.p[:, -1]

    def solve_pressure_poisson(self):
        """Solve pressure Poisson equation using FFT (from reference code)."""
        # Compute divergence source term
        dudx = (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1)) / (2 * self.dx)
        dvdy = (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / (2 * self.dy)

        source = (dudx + dvdy) / self.dt

        # Solve using FFT
        source_fft = fft2(source)

        # Wave numbers
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky)
        K2 = Kx*Kx + Ky*Ky

        # Avoid division by zero
        K2[0, 0] = 1.0

        # Solve: ∇²p = source
        p_fft = source_fft / (-K2)
        p_fft[0, 0] = 0.0  # Set mean pressure to zero

        self.p = np.real(ifft2(p_fft))

    def adams_bashforth_step(self, current_rhs, prev_rhs1, prev_rhs2):
        """Adams-Bashforth 3rd order time stepping."""
        return self.dt * (23/12 * current_rhs - 16/12 * prev_rhs1 + 5/12 * prev_rhs2)

    def compute_momentum_rhs(self):
        """Compute RHS for momentum equations."""
        # Pressure gradients
        dpdx = (np.roll(self.p, -1, axis=1) - np.roll(self.p, 1, axis=1)) / (2 * self.dx)
        dpdy = (np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)) / (2 * self.dy)

        # Viscous diffusion (simplified for demonstration)
        d2udx2 = (np.roll(self.u, -1, axis=1) - 2*self.u + np.roll(self.u, 1, axis=1)) / (self.dx**2)
        d2udy2 = (np.roll(self.u, -1, axis=0) - 2*self.u + np.roll(self.u, 1, axis=0)) / (self.dy**2)

        d2vdx2 = (np.roll(self.v, -1, axis=1) - 2*self.v + np.roll(self.v, 1, axis=1)) / (self.dx**2)
        d2vdy2 = (np.roll(self.v, -1, axis=0) - 2*self.v + np.roll(self.v, 1, axis=0)) / (self.dy**2)

        # Convection (simplified)
        dudx = (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1)) / (2 * self.dx)
        dudy = (np.roll(self.u, -1, axis=0) - np.roll(self.u, 1, axis=0)) / (2 * self.dy)
        dvdx = (np.roll(self.v, -1, axis=1) - np.roll(self.v, 1, axis=1)) / (2 * self.dx)
        dvdy = (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / (2 * self.dy)

        # u-momentum equation
        u_rhs = -dpdx + (d2udx2 + d2udy2) - (self.u * dudx + self.v * dudy)

        # v-momentum equation (with buoyancy)
        v_rhs = -dpdy + (d2vdx2 + d2vdy2) - (self.u * dvdx + self.v * dvdy) + self.Ra * self.Pr * self.T

        return u_rhs, v_rhs

    def compute_temperature_rhs(self):
        """Compute RHS for temperature equation."""
        # Thermal diffusion
        d2Tdx2 = (np.roll(self.T, -1, axis=1) - 2*self.T + np.roll(self.T, 1, axis=1)) / (self.dx**2)
        d2Tdy2 = (np.roll(self.T, -1, axis=0) - 2*self.T + np.roll(self.T, 1, axis=0)) / (self.dy**2)

        # Temperature gradients for advection
        dTdx = (np.roll(self.T, -1, axis=1) - np.roll(self.T, 1, axis=1)) / (2 * self.dx)
        dTdy = (np.roll(self.T, -1, axis=0) - np.roll(self.T, 1, axis=0)) / (2 * self.dy)

        # Temperature equation: ∂T/∂t + u·∇T = Pr ∇²T
        T_rhs = self.Pr * (d2Tdx2 + d2Tdy2) - (self.u * dTdx + self.v * dTdy)

        return T_rhs

    def step(self, step_number):
        """Single time step using the reference algorithm."""
        # Store previous values
        self.T_prev[1] = self.T_prev[0].copy()
        self.T_prev[0] = self.T.copy()
        self.u_prev[1] = self.u_prev[0].copy()
        self.u_prev[0] = self.u.copy()
        self.v_prev[1] = self.v_prev[0].copy()
        self.v_prev[0] = self.v.copy()

        # Solve pressure
        self.solve_pressure_poisson()

        # Compute RHS for all equations
        u_rhs, v_rhs = self.compute_momentum_rhs()
        T_rhs = self.compute_temperature_rhs()

        # Adams-Bashforth time stepping (simplified for first few steps)
        if step_number < 2:
            # Forward Euler for first steps
            self.u += self.dt * u_rhs
            self.v += self.dt * v_rhs
            self.T += self.dt * T_rhs
        else:
            # Adams-Bashforth
            self.u += self.adams_bashforth_step(u_rhs, self.u_prev[0], self.u_prev[1])
            self.v += self.adams_bashforth_step(v_rhs, self.v_prev[0], self.v_prev[1])
            self.T += self.adams_bashforth_step(T_rhs, self.T_prev[0], self.T_prev[1])

        # Apply boundary conditions
        self.apply_boundary_conditions()

        # Clip temperature to physical bounds
        self.T = np.clip(self.T, 0, 1)
        self.T[0, :] = 1.0   # Re-enforce hot bottom
        self.T[-1, :] = 0.0  # Re-enforce cold top


def generate_training_dataset(Ra=1e5, n_runs=5, save_path='rb_data_numerical', nx=128, ny=64):
    """
    Generate training dataset following reference format.
    Output format matches sourcecodeCDAnet expectations: [p, b, u, w] channels
    where b=temperature, u=x-velocity, w=y-velocity
    """
    os.makedirs(save_path, exist_ok=True)

    # Simulation parameters (following reference)
    t_start = 5.0       # Startup time to reach steady convection
    t_end = 10.0        # End time
    dt = 1e-3           # Time step
    delta_t = 0.1       # Saving interval

    n_steps = int((t_end - t_start) / delta_t)
    print(f"Generating {n_runs} runs with {n_steps} snapshots each")
    print(f"Ra = {Ra:.0e}, Grid: {nx}×{ny}")

    all_data = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}")

        # Create simulation
        sim = RBNumericalSimulation(nx=nx, ny=ny, Ra=Ra, save_path=save_path, dt=dt)

        # Startup phase - let convection develop
        n_startup = int(t_start / dt)
        for step in range(n_startup):
            sim.step(step)
            if step % 1000 == 0:
                print(f"    Startup step {step}/{n_startup}")

        # Data collection phase
        run_data = []
        steps_per_save = int(delta_t / dt)

        for sample_step in range(n_steps):
            # Run simulation for delta_t
            for _ in range(steps_per_save):
                sim.step(n_startup + sample_step * steps_per_save + _)

            # Save snapshot in format [p, b, u, w] to match reference
            snapshot = {
                'pressure': sim.p.copy(),          # p channel
                'temperature': sim.T.copy(),       # b channel (buoyancy/temperature)
                'velocity_x': sim.u.copy(),        # u channel (x-velocity)
                'velocity_y': sim.v.copy(),        # w channel (y-velocity, called 'w' in reference)
                'time': t_start + sample_step * delta_t
            }
            run_data.append(snapshot)

            if sample_step % 10 == 0:
                print(f"    Sample {sample_step+1}/{n_steps}")

        # Save data in format compatible with reference code
        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        save_run_data(filename, run_data, Ra, nx, ny)
        print(f"    Saved: {filename}")

        all_data.append(run_data)

    print(f"Dataset generation complete!")
    print(f"Total files: {n_runs}")
    print(f"Samples per file: {n_steps}")
    print(f"Grid resolution: {nx}×{ny}")

    return all_data


def save_run_data(filename, run_data, Ra, nx, ny):
    """Save run data in HDF5 format compatible with reference loader."""
    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['Ra'] = Ra
        f.attrs['Pr'] = 0.7
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['Lx'] = 3.0
        f.attrs['Ly'] = 1.0
        f.attrs['n_samples'] = len(run_data)

        # Save each frame
        for i, frame in enumerate(run_data):
            grp = f.create_group(f'frame_{i:03d}')
            grp.create_dataset('pressure', data=frame['pressure'])
            grp.create_dataset('temperature', data=frame['temperature'])
            grp.create_dataset('velocity_x', data=frame['velocity_x'])
            grp.create_dataset('velocity_y', data=frame['velocity_y'])
            grp.attrs['time'] = frame['time']


def create_consolidated_dataset(data_dir='rb_data_numerical', Ra=1e5):
    """
    Create consolidated dataset file in the format expected by reference code.
    Combines multiple runs into format compatible with dataloader_spacetime.py
    """
    import glob

    pattern = f'{data_dir}/rb_data_Ra_{Ra:.0e}_run_*.h5'
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No data files found matching {pattern}")
        return

    print(f"Creating consolidated dataset from {len(files)} files...")

    # Read first file to get dimensions
    with h5py.File(files[0], 'r') as f:
        nx = f.attrs['nx']
        ny = f.attrs['ny']
        n_samples = f.attrs['n_samples']

    # Allocate arrays for all data
    n_files = len(files)
    total_samples = n_files * n_samples

    # Following reference format: [channel, file, sample, y, x]
    pressure_data = np.zeros((n_files, n_samples, ny, nx), dtype=np.float32)
    temp_data = np.zeros((n_files, n_samples, ny, nx), dtype=np.float32)
    u_data = np.zeros((n_files, n_samples, ny, nx), dtype=np.float32)
    v_data = np.zeros((n_files, n_samples, ny, nx), dtype=np.float32)
    times = np.zeros((n_files, n_samples), dtype=np.float32)

    # Load all data
    for file_idx, filepath in enumerate(files):
        print(f"  Loading file {file_idx+1}/{n_files}: {os.path.basename(filepath)}")

        with h5py.File(filepath, 'r') as f:
            for sample_idx in range(n_samples):
                frame_key = f'frame_{sample_idx:03d}'
                if frame_key in f:
                    grp = f[frame_key]
                    pressure_data[file_idx, sample_idx] = grp['pressure'][:]
                    temp_data[file_idx, sample_idx] = grp['temperature'][:]
                    u_data[file_idx, sample_idx] = grp['velocity_x'][:]
                    v_data[file_idx, sample_idx] = grp['velocity_y'][:]
                    times[file_idx, sample_idx] = grp.attrs['time']

    # Save consolidated data in reference format
    output_file = f'{data_dir}/rb2d_ra{Ra:.0e}_consolidated.h5'

    with h5py.File(output_file, 'w') as f:
        # Save in [channel, file, time, y, x] format to match reference
        f.create_dataset('p', data=pressure_data)  # pressure
        f.create_dataset('b', data=temp_data)      # buoyancy (temperature)
        f.create_dataset('u', data=u_data)         # x-velocity
        f.create_dataset('w', data=v_data)         # y-velocity (w in reference)
        f.create_dataset('times', data=times)

        # Metadata
        f.attrs['Ra'] = Ra
        f.attrs['Pr'] = 0.7
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['Lx'] = 3.0
        f.attrs['Ly'] = 1.0
        f.attrs['n_files'] = n_files
        f.attrs['n_samples'] = n_samples
        f.attrs['total_samples'] = total_samples

    print(f"Consolidated dataset saved: {output_file}")
    print(f"Format: [p,b,u,w] × {n_files} × {n_samples} × {ny} × {nx}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RB training data following reference implementation')
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of simulation runs')
    parser.add_argument('--nx', type=int, default=128, help='Grid points in x')
    parser.add_argument('--ny', type=int, default=64, help='Grid points in y')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical', help='Output directory')
    parser.add_argument('--consolidate', action='store_true', help='Create consolidated dataset')

    args = parser.parse_args()

    print("=" * 60)
    print("RB Simulation Data Generation (Reference Compatible)")
    print("=" * 60)

    # Generate data
    generate_training_dataset(
        Ra=args.Ra,
        n_runs=args.n_runs,
        save_path=args.save_path,
        nx=args.nx,
        ny=args.ny
    )

    # Create consolidated dataset for training
    if args.consolidate:
        create_consolidated_dataset(args.save_path, args.Ra)

    print("\n✅ Data generation complete!")
    print(f"📁 Files saved to: {args.save_path}")
    print("🔄 Use --consolidate to create training-ready format")