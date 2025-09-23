#!/usr/bin/env python3
"""
Simplified but correct RB simulation based on sourcecodeCDAnet reference.
This version uses the exact same approach as the reference code.
"""

import numpy as np
from scipy.fft import fft2, ifft2
import h5py
import os
import argparse


class RBSimulation:
    """
    Simplified RB simulation matching sourcecodeCDAnet/sim.py exactly.
    """

    def __init__(self, nx=128, ny=64, Lx=3.0, Ly=1.0, Ra=1e5, Pr=0.7, dt=1e-3):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.Ra = Ra
        self.Pr = Pr
        self.dt = dt

        self.dx = Lx / nx
        self.dy = Ly / ny

        # Initialize fields [ny, nx] - y first, x second
        self.T = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        # Previous time steps for Adams-Bashforth
        self.T_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.u_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.v_prev = [np.zeros((ny, nx)) for _ in range(2)]

        self.setup_initial_conditions()

    def setup_initial_conditions(self):
        """Setup initial conditions exactly as in reference."""
        # Temperature profile: hot bottom (T=1), cold top (T=0)
        self.T[-1, :] = 1.0  # Bottom wall (max y index)
        self.T[0, :] = 0.0   # Top wall (min y index)

        # Add small perturbations
        self.T += 1e-3 * np.random.randn(self.ny, self.nx)

        # Initialize history
        for i in range(2):
            self.T_prev[i] = self.T.copy()
            self.u_prev[i] = self.u.copy()
            self.v_prev[i] = self.v.copy()

    def solve_pressure_poisson(self):
        """Solve pressure equation from reference code."""
        # Compute divergence source - exactly from reference
        source = (
            (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1)) / (2*self.dx) +
            (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / (2*self.dy)
        ) / self.dt

        # FFT solution - exactly from reference
        source_fft = fft2(source)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky)
        K2 = Kx*Kx + Ky*Ky
        K2[0,0] = 1.0  # Avoid division by zero

        p_fft = source_fft / (-K2)
        self.p = np.real(ifft2(p_fft))

    def adams_bashforth_step(self, f, f_prev1, f_prev2):
        """Adams-Bashforth step - exactly from reference."""
        return f + self.dt * (
            23/12 * f_prev1 - 16/12 * f_prev2 + 5/12 * f
        )

    def step(self, step_number):
        """Single time step - exactly from reference."""
        # Store previous values
        self.T_prev[1] = self.T_prev[0].copy()
        self.T_prev[0] = self.T.copy()
        self.u_prev[1] = self.u_prev[0].copy()
        self.u_prev[0] = self.u.copy()
        self.v_prev[1] = self.v_prev[0].copy()
        self.v_prev[0] = self.v.copy()

        # Solve pressure
        self.solve_pressure_poisson()

        # Momentum equations - exactly from reference
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

        # Temperature equation - exactly from reference
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

        # Apply constraints - exactly from reference
        self.T = np.clip(T_update, 0, 1)
        self.T[0, :] = 0.0   # Cold top
        self.T[-1, :] = 1.0  # Hot bottom

        # Velocity boundary conditions - exactly from reference
        self.u[:, 0] = self.u[:, -1]  # Periodic in x
        self.u[:, -1] = self.u[:, 0]
        self.v[:, 0] = self.v[:, -1]  # Periodic in x
        self.v[:, -1] = self.v[:, 0]


def generate_training_dataset(Ra=1e5, n_runs=5, save_path='rb_data_numerical'):
    """Generate dataset exactly matching reference format."""
    os.makedirs(save_path, exist_ok=True)

    # Parameters exactly from reference
    t_start = 5
    t_end = 10
    dt = 1e-3
    delta_t = 0.1

    n_steps = int((t_end - t_start) / delta_t)

    print(f"Generating {n_runs} runs with {n_steps} snapshots each")
    print(f"Ra = {Ra:.0e}")

    all_data = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}")

        sim = RBSimulation(Ra=Ra, dt=dt)

        # Startup phase
        n_startup = int(t_start / dt)
        for step in range(n_startup):
            sim.step(step)
            if step % 1000 == 0:
                print(f"    Startup: {step}/{n_startup}")

        # Data collection
        data = []
        steps_per_save = int(delta_t / dt)

        for step_number in range(n_steps):
            for _ in range(steps_per_save):
                sim.step(step_number)

            # Save in exact reference format
            data.append({
                'pressure': sim.p.copy(),     # p channel
                'temperature': sim.T.copy(),  # b channel (buoyancy)
                'velocity_x': sim.u.copy(),   # u channel
                'velocity_y': sim.v.copy()    # w channel
            })

            if step_number % 10 == 0:
                print(f"    Sample: {step_number+1}/{n_steps}")

        # Save individual run file
        filename = f'{save_path}/ra_{Ra}_run_{run}.h5'
        with h5py.File(filename, 'w') as f:
            for i, frame in enumerate(data):
                grp = f.create_group(f'frame_{i}')
                for key, value in frame.items():
                    grp.create_dataset(key, data=value)

        print(f"    Saved: {filename}")
        all_data.append(data)

    # Create consolidated format for training
    create_consolidated_dataset(save_path, Ra, all_data)

    return all_data


def create_consolidated_dataset(save_path, Ra, all_data):
    """Create consolidated dataset in reference format."""
    n_runs = len(all_data)
    n_samples = len(all_data[0])
    ny, nx = all_data[0][0]['temperature'].shape

    print(f"Creating consolidated dataset: {n_runs} runs × {n_samples} samples × {ny}×{nx}")

    # Allocate arrays in reference format: [run, sample, y, x]
    p_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    b_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)  # temperature (buoyancy)
    u_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    w_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)  # v-velocity (called w in ref)

    for run_idx, run_data in enumerate(all_data):
        for sample_idx, frame in enumerate(run_data):
            p_data[run_idx, sample_idx] = frame['pressure']
            b_data[run_idx, sample_idx] = frame['temperature']  # b = buoyancy/temperature
            u_data[run_idx, sample_idx] = frame['velocity_x']
            w_data[run_idx, sample_idx] = frame['velocity_y']   # w = v-velocity

    # Save in exact reference format
    output_file = f'{save_path}/rb2d_ra{Ra:.0e}_consolidated.h5'

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('p', data=p_data)
        f.create_dataset('b', data=b_data)  # buoyancy (temperature)
        f.create_dataset('u', data=u_data)
        f.create_dataset('w', data=w_data)  # w instead of v

        # Metadata
        f.attrs['Ra'] = Ra
        f.attrs['Pr'] = 0.7
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['Lx'] = 3.0
        f.attrs['Ly'] = 1.0
        f.attrs['n_runs'] = n_runs
        f.attrs['n_samples'] = n_samples

    print(f"✅ Consolidated dataset: {output_file}")
    print(f"   Format: [p,b,u,w] × {n_runs} × {n_samples} × {ny} × {nx}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple RB data generation')
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical', help='Output directory')

    args = parser.parse_args()

    print("=" * 60)
    print("Simple RB Data Generation (Reference Compatible)")
    print("=" * 60)

    generate_training_dataset(Ra=args.Ra, n_runs=args.n_runs, save_path=args.save_path)

    print("\n✅ Dataset generation complete!")
    print(f"📁 Files saved to: {args.save_path}")