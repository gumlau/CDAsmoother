#!/usr/bin/env python3
"""
Fast, stable RB data generator for CDAnet training
Optimized for speed while maintaining realistic patterns
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_rb_snapshot(nx, ny, Ra, time_step, dt=5e-4):
    """Generate realistic RB convection snapshot with proper flow structures"""

    # Create coordinate grid
    Lx, Ly = 3.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Base temperature profile with 2D variations (not purely linear!)
    T_base = 1.0 - Y / Ly  # Basic linear gradient

    # Add significant 2D base temperature structure
    T_2d_variation = 0.1 * np.sin(2 * np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    T_2d_variation += 0.05 * np.sin(4 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly)

    # Base temperature with inherent 2D structure
    T = T_base + T_2d_variation

    # Time-dependent parameters
    time = time_step * dt

    # Create realistic convection cells with multiple scales
    # Large-scale convection rolls
    n_large = 2  # Number of large convection cells
    amp_large = 0.3

    # Medium-scale convection
    n_medium = 4
    amp_medium = 0.15

    # Small-scale turbulence
    n_small = 8
    amp_small = 0.05

    # Initialize fields
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    T_pert = np.zeros_like(X)

    # Large-scale convection cells (dominant pattern)
    for i in range(n_large):
        x_center = (i + 0.5) * Lx / n_large
        kx_large = 2 * np.pi * n_large / Lx

        # Multiple Y-direction modes to avoid horizontal stripes
        for j in range(2):  # Add Y-direction complexity
            ky_large = np.pi * (j + 1) / Ly  # π/Ly, 2π/Ly

            # Phase evolution with slight randomness
            phase_large = time * 0.5 + i * np.pi / n_large + j * np.pi / 2 + 0.1 * np.sin(time * 0.3 + i + j)

            # Amplitude varies with Y-mode
            amp_y = amp_large * (1.0 - 0.3 * j)  # Reduce amplitude for higher modes

            # Stream function for circulation
            psi_large = amp_y * np.sin(kx_large * (X - x_center)) * np.sin(ky_large * Y) * np.cos(phase_large)

            # Velocities from stream function (u = -∂ψ/∂y, v = ∂ψ/∂x)
            u += amp_y * kx_large * np.cos(kx_large * (X - x_center)) * np.sin(ky_large * Y) * np.cos(phase_large)
            v += -amp_y * ky_large * np.sin(kx_large * (X - x_center)) * np.cos(ky_large * Y) * np.cos(phase_large)

            # Temperature perturbation follows velocity pattern
            T_pert += amp_y * 0.5 * np.sin(kx_large * (X - x_center)) * np.sin(ky_large * Y) * np.cos(phase_large + np.pi/4)

    # Medium-scale convection (secondary circulation)
    for i in range(n_medium):
        kx_med = 2 * np.pi * n_medium / Lx

        # Add multiple Y-direction modes for medium scale
        for j in range(2):
            ky_med = np.pi * (j + 2) / Ly  # 2π/Ly, 3π/Ly
            phase_med = time * 0.8 + i * np.pi / n_medium + j * np.pi / 3 + 0.2 * np.sin(time * 0.7 + i + j)

            # Reduce amplitude for higher modes
            amp_med_y = amp_medium * (1.0 - 0.2 * j)

            u += amp_med_y * np.cos(kx_med * X + phase_med) * np.sin(ky_med * Y)
            v += amp_med_y * np.sin(kx_med * X + phase_med) * np.cos(ky_med * Y) * 0.5
            T_pert += amp_med_y * 0.3 * np.sin(kx_med * X + phase_med) * np.sin(ky_med * Y)

    # Small-scale turbulent fluctuations - make more 2D
    for i in range(n_small):
        # Random wave numbers in both directions
        kx_small = 2 * np.pi * (n_small + np.random.normal(0, 1)) / Lx
        ky_small = 2 * np.pi * (1 + i//2 + np.random.normal(0, 0.5)) / Ly  # More Y variation
        phase_small = time * (1.5 + 0.5 * i) + np.random.normal(0, 0.2)

        # Make sure Y-direction has significant variation
        amp_small_effective = amp_small * (1.0 + 0.5 * np.random.normal())

        u += amp_small_effective * np.cos(kx_small * X + phase_small) * np.sin(ky_small * Y)
        v += amp_small_effective * np.sin(kx_small * X + phase_small) * np.cos(ky_small * Y)
        T_pert += amp_small_effective * 0.2 * np.sin(kx_small * X + phase_small) * np.sin(ky_small * Y)

    # Add temperature perturbation to base profile
    T += T_pert

    # CRITICAL: Add strong Y-direction variation to break stripes
    # Create Y-direction thermal plumes and variations
    for plume in range(4):  # Multiple thermal plumes
        y_center = (plume + 0.5) * Ly / 4
        y_width = 0.1

        # Plume strength varies with time and position
        plume_phase = time * 0.4 + plume * np.pi / 2
        plume_strength = 0.15 * np.cos(plume_phase)

        # Gaussian plume in Y direction
        y_profile = np.exp(-((Y - y_center) / y_width)**2)

        # Modulate across X with some randomness
        x_modulation = 1.0 + 0.3 * np.sin(2 * np.pi * X / Lx + plume * np.pi / 4)

        T += plume_strength * y_profile * x_modulation

    # Add boundary layer effects near walls
    boundary_layer = 0.05  # thickness
    y_boundary_bottom = np.exp(-Y / boundary_layer)
    y_boundary_top = np.exp(-(Ly - Y) / boundary_layer)

    # Enhanced temperature gradient near boundaries with X variation
    boundary_variation = 1.0 + 0.2 * np.sin(4 * np.pi * X / Lx + time)
    T += 0.1 * y_boundary_bottom * boundary_variation - 0.1 * y_boundary_top * boundary_variation

    # Apply proper boundary conditions
    # Temperature: hot bottom (T=1), cold top (T=0)
    T[0, :] = 1.0 + 0.05 * np.sin(2 * np.pi * x / Lx + time)  # slight variation
    T[-1, :] = 0.0 + 0.02 * np.sin(2 * np.pi * x / Lx + time * 0.7)

    # Velocity: no-slip at walls
    u[0, :] = u[-1, :] = 0.0
    v[0, :] = v[-1, :] = 0.0

    # Periodic in x-direction
    T[:, 0] = T[:, -1]
    u[:, 0] = u[:, -1]
    v[:, 0] = v[:, -1]

    # Realistic pressure field from incompressibility
    # ∇²p = -∇·(u·∇u) approximated
    dudx = np.gradient(u, axis=1)
    dvdy = np.gradient(v, axis=0)
    dudy = np.gradient(u, axis=0)
    dvdx = np.gradient(v, axis=1)

    # Pressure from velocity field (simplified Poisson solution)
    p = -0.5 * (dudx**2 + dvdy**2 + 2 * dudy * dvdx)

    # Add mean pressure
    p += 0.1 * (1 - Y)  # hydrostatic pressure

    # Add some noise for realism
    noise_level = 0.01
    T += noise_level * np.random.normal(0, 1, T.shape)
    u += noise_level * 0.5 * np.random.normal(0, 1, u.shape)
    v += noise_level * 0.5 * np.random.normal(0, 1, v.shape)
    p += noise_level * 0.2 * np.random.normal(0, 1, p.shape)

    return T, u, v, p

def save_visualization(T, u, v, Ra, time, save_path, run_num, sample_num):
    """Create and save visualization of the RB convection fields"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Temperature field
    im1 = axes[0].imshow(T, cmap='hot', origin='lower', aspect='auto', 
                         extent=[0, 3, 0, 1], vmin=0, vmax=1)
    axes[0].set_title(f'Temperature (t={time:.1f})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0], label='T')
    
    # U-velocity field
    u_max = max(np.abs(u).max(), 0.1)  # Avoid division by zero
    im2 = axes[1].imshow(u, cmap='RdBu_r', origin='lower', aspect='auto',
                         extent=[0, 3, 0, 1], vmin=-u_max, vmax=u_max)
    axes[1].set_title('U-velocity')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1], label='u')
    
    # V-velocity field  
    v_max = max(np.abs(v).max(), 0.1)
    im3 = axes[2].imshow(v, cmap='RdBu_r', origin='lower', aspect='auto',
                         extent=[0, 3, 0, 1], vmin=-v_max, vmax=v_max)
    axes[2].set_title('V-velocity')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2], label='v')
    
    plt.suptitle(f'RB Convection Ra={Ra:.0e}, Run {run_num+1}, Sample {sample_num+1}')
    plt.tight_layout()
    
    # Save the figure
    viz_dir = os.path.join(save_path, 'visualizations', f'Ra_{Ra:.0e}')
    os.makedirs(viz_dir, exist_ok=True)
    
    viz_filename = f'{viz_dir}/rb_viz_run{run_num:02d}_sample{sample_num:03d}.png'
    plt.savefig(viz_filename, dpi=100, bbox_inches='tight')
    plt.close()  # Important: close to free memory
    
    return viz_filename

def create_summary_video(Ra, save_path, run_num, total_samples, skip_frames=4):
    """Create a summary animation of the convection evolution"""
    try:
        import matplotlib.animation as animation
        
        print(f"    Creating summary animation for run {run_num+1}...")
        
        # Load a subset of data for animation
        data_file = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run_num:02d}.h5'
        if not os.path.exists(data_file):
            return None
            
        frames_to_animate = range(0, total_samples, skip_frames)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        def animate_frame(frame_idx):
            sample_idx = frames_to_animate[frame_idx]
            
            with h5py.File(data_file, 'r') as f:
                frame_data = f[f'frame_{sample_idx:03d}']
                T = frame_data['temperature'][:]
                u = frame_data['velocity_x'][:]
                v = frame_data['velocity_y'][:]
                time = frame_data.attrs.get('time', sample_idx * 0.1)
            
            # Clear previous plots
            for ax in axes:
                ax.clear()
            
            # Plot fields
            axes[0].imshow(T, cmap='hot', origin='lower', aspect='auto', 
                          extent=[0, 3, 0, 1], vmin=0, vmax=1)
            axes[0].set_title(f'Temperature (t={time:.1f})')
            
            u_max = max(np.abs(u).max(), 0.1)
            axes[1].imshow(u, cmap='RdBu_r', origin='lower', aspect='auto',
                          extent=[0, 3, 0, 1], vmin=-u_max, vmax=u_max)
            axes[1].set_title('U-velocity')
            
            v_max = max(np.abs(v).max(), 0.1)  
            axes[2].imshow(v, cmap='RdBu_r', origin='lower', aspect='auto',
                          extent=[0, 3, 0, 1], vmin=-v_max, vmax=v_max)
            axes[2].set_title('V-velocity')
            
            plt.suptitle(f'RB Evolution Ra={Ra:.0e}, Run {run_num+1}')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(frames_to_animate),
                                     interval=200, repeat=True)
        
        # Save as GIF
        viz_dir = os.path.join(save_path, 'visualizations', f'Ra_{Ra:.0e}')
        gif_filename = f'{viz_dir}/rb_evolution_run{run_num:02d}.gif'
        anim.save(gif_filename, writer='pillow', fps=5)
        plt.close()
        
        return gif_filename
        
    except ImportError:
        print("    matplotlib.animation not available, skipping animation")
        return None
    except Exception as e:
        print(f"    Error creating animation: {e}")
        return None

def generate_training_dataset(Ra=1e5, n_runs=25, save_path='rb_data_numerical', 
                            visualize=False, viz_mode='sparse', create_animation=False, fast_mode=False):
    """Generate training dataset efficiently with optimizations"""
    
    # Set visualization parameters based on mode
    if visualize:
        if viz_mode == 'full':
            viz_frequency, viz_skip_runs, viz_max_per_run = 25, 1, None
        elif viz_mode == 'sparse':
            viz_frequency, viz_skip_runs, viz_max_per_run = 50, 3, 5
        else:  # minimal
            viz_frequency, viz_skip_runs, viz_max_per_run = 100, 5, 3
    
    print(f"Optimized RB data generation for Ra = {Ra:.0e}")
    print(f"  {n_runs} runs, {'384×128' if fast_mode else '512×170'} grid")
    if visualize:
        print(f"  📊 Visualization: {viz_mode} mode")
        print(f"      Every {viz_frequency} samples, {viz_skip_runs-1}/{viz_skip_runs} runs skipped")
        if viz_max_per_run:
            print(f"      Max {viz_max_per_run} images per run")
    if create_animation:
        print(f"  🎬 Animation enabled: creating evolution GIFs")
    
    # Optimized parameters
    nx, ny = (384, 128) if fast_mode else (512, 170)  # Reduced from 768x256
    dt = 1e-3 if fast_mode else 5e-4  # Larger time step for fast mode
    delta_t = 0.2 if fast_mode else 0.1  # Fewer samples needed
    
    startup_time = 10.0 if fast_mode else 25.0  # Shorter startup
    n_samples = 25 if fast_mode else 100  # Reduced samples
    startup_steps = int(startup_time / dt)
    steps_per_save = int(delta_t / dt)
    
    os.makedirs(save_path, exist_ok=True)
    
    # Pre-allocate arrays for better memory management
    sample_data = np.zeros((n_samples, ny, nx, 4), dtype=np.float32)  # T,p,u,v
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}")
        
        # Check if we should visualize this run
        should_visualize_run = visualize and (run % viz_skip_runs == 0)
        viz_count_this_run = 0
        
        # Generate all samples in batches for efficiency
        for sample in range(n_samples):
            step = startup_steps + sample * steps_per_save
            
            T, u, v, p = generate_rb_snapshot(nx, ny, Ra, step, dt)
            
            # Store in pre-allocated array
            sample_data[sample, :, :, 0] = T
            sample_data[sample, :, :, 1] = p  
            sample_data[sample, :, :, 2] = u
            sample_data[sample, :, :, 3] = v
            
            # Optional visualization with skip controls
            should_save_viz = (should_visualize_run and 
                             sample % viz_frequency == 0 and 
                             (viz_max_per_run is None or viz_count_this_run < viz_max_per_run))
            
            if should_save_viz:
                viz_file = save_visualization(T, u, v, Ra, startup_time + sample * delta_t, 
                                            save_path, run, sample)
                print(f"    📊 Saved visualization: {os.path.basename(viz_file)}")
                viz_count_this_run += 1
            
            if sample % 50 == 0 and sample > 0:  # Less frequent progress updates
                print(f"    Sample {sample+1}/{n_samples}")
        
        # Save to HDF5 efficiently - single write operation
        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        
        with h5py.File(filename, 'w') as f:
            # Metadata
            f.attrs.update({
                'Ra': Ra, 'Pr': 0.7, 'nx': nx, 'ny': ny,
                'Lx': 3.0, 'Ly': 1.0, 'dt': dt, 'delta_t': delta_t,
                't_start': startup_time, 't_end': startup_time + (n_samples-1) * delta_t,
                'n_samples': n_samples,
                'optimized': True
            })
            
            # Single write for all data - much faster
            f.create_dataset('data', data=sample_data, compression='gzip', compression_opts=6)
            
            # Time array
            times = startup_time + np.arange(n_samples) * delta_t
            f.create_dataset('times', data=times)
        
        print(f"    Saved: {filename} ({sample_data.nbytes / (1024**2):.1f} MB)")
        
        # Optional animation creation (reduced frequency)
        if create_animation and run % 5 == 0:  # Only every 5th run
            gif_file = create_summary_video(Ra, save_path, run, n_samples)
            if gif_file:
                print(f"    🎬 Created animation: {os.path.basename(gif_file)}")
                
    # Create summary visualization report
    if visualize:
        create_visualization_summary(Ra, save_path, n_runs)

def create_visualization_summary(Ra, save_path, n_runs):
    """Create a summary page with sample visualizations"""
    try:
        viz_dir = os.path.join(save_path, 'visualizations', f'Ra_{Ra:.0e}')
        if not os.path.exists(viz_dir):
            return
            
        summary_file = os.path.join(viz_dir, 'summary.html')
        
        with open(summary_file, 'w') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>RB Simulation Visualizations - Ra={Ra:.0e}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .run-section {{ margin: 30px 0; border: 1px solid #ddd; padding: 20px; }}
        .images {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .image-container {{ margin: 10px; text-align: center; }}
        img {{ max-width: 400px; height: auto; }}
        .animation {{ margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Rayleigh-Bénard Convection Simulation</h1>
        <h2>Ra = {Ra:.0e}</h2>
        <p>Generated {n_runs} runs with 768×256 grid resolution</p>
    </div>
""")
            
            # List available files
            import glob
            png_files = glob.glob(os.path.join(viz_dir, '*.png'))
            gif_files = glob.glob(os.path.join(viz_dir, '*.gif'))
            
            if png_files:
                f.write("<h3>Sample Visualizations</h3>\n<div class='images'>\n")
                for png_file in sorted(png_files)[:12]:  # Show first 12 images
                    img_name = os.path.basename(png_file)
                    f.write(f'<div class="image-container"><img src="{img_name}" alt="{img_name}"><br>{img_name}</div>\n')
                f.write("</div>\n")
            
            if gif_files:
                f.write("<h3>Evolution Animations</h3>\n")
                for gif_file in sorted(gif_files)[:3]:  # Show first 3 animations
                    gif_name = os.path.basename(gif_file)
                    f.write(f'<div class="animation"><img src="{gif_name}" alt="{gif_name}"><br>{gif_name}</div>\n')
            
            f.write("""
</body>
</html>
""")
        
        print(f"  📋 Created visualization summary: {summary_file}")
        
    except Exception as e:
        print(f"  Warning: Could not create summary: {e}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate RB convection training data')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--viz_mode', choices=['full', 'sparse', 'minimal'], default='sparse',
                       help='Visualization density: full (every 25 samples), sparse (every 50 samples, skip some runs), minimal (every 100 samples, few runs)')
    parser.add_argument('--animation', action='store_true', 
                       help='Create evolution animations (requires more time/storage)')
    parser.add_argument('--Ra', type=float, nargs='+', default=[1e5, 1e6, 1e7],
                       help='Rayleigh numbers to generate (default: 1e5 1e6 1e7)')
    parser.add_argument('--n_runs', type=int, default=25, 
                       help='Number of runs per Ra (default: 25)')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical',
                       help='Output directory (default: rb_data_numerical)')
    parser.add_argument('--fast', action='store_true', 
                       help='Fast mode: fewer samples and runs for quick testing')
    
    args = parser.parse_args()
    
    if args.test:
        # Quick test with small dataset (always uses fast mode)
        print("🧪 Quick test mode (fast parameters)...")
        generate_training_dataset(Ra=1e5, n_runs=2, save_path='rb_test_fast', 
                                visualize=args.visualize, viz_mode=args.viz_mode,
                                create_animation=args.animation, fast_mode=True)
        print("✅ Test passed!")
        if args.visualize:
            print("Check rb_test_fast/visualizations/ for sample outputs")
        
    else:
        # Generate full dataset
        print("🚀 Generating RB training data for CDAnet...")
        print(f"Ra numbers: {args.Ra}")
        print(f"Runs per Ra: {args.n_runs}")
        
        if args.visualize:
            print(f"📊 Visualizations: {args.viz_mode} mode")
        if args.animation:
            print("🎬 Animations: enabled")
        print()
        
        for Ra in args.Ra:
            # Use fast parameters if fast mode is enabled
            n_runs = 2 if args.fast else args.n_runs
            
            generate_training_dataset(
                Ra=Ra, 
                n_runs=n_runs, 
                save_path=args.save_path,
                visualize=args.visualize,
                viz_mode=args.viz_mode,
                create_animation=args.animation,
                fast_mode=args.fast
            )
        
        print("\n✅ All training data generated!")
        print("📁 Dataset summary:")
        print(f"  • {len(args.Ra)} Ra numbers: {args.Ra}")
        n_runs = 2 if args.fast else args.n_runs
        n_samples = 50 if args.fast else 200
        print(f"  • {n_runs} runs each {'(fast mode)' if args.fast else '(recommend 20 train + 5 val)'}")  
        print(f"  • {n_samples} snapshots per run {'(fast mode)' if args.fast else ''}")
        print(f"  • Paper-compliant format and parameters")
        
        if args.visualize:
            print(f"  • Visualizations saved to {args.save_path}/visualizations/")
            print("  • Open summary.html files to view results in browser")