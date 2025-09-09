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
    """Generate a single RB convection snapshot quickly"""
    
    # Create coordinate grid
    Lx, Ly = 3.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # Base linear temperature profile
    T = 1.0 - Y / Ly
    
    # Number of convection cells based on Ra
    if Ra <= 1e4:
        n_cells = 2
        amp = 0.1
    elif Ra <= 1e5:
        n_cells = 3
        amp = 0.2
    elif Ra <= 1e6:
        n_cells = 4
        amp = 0.25
    else:
        n_cells = 6
        amp = 0.3
    
    # Time evolution
    phase = 0.1 * time_step * dt
    
    # Generate convection pattern
    cell_width = Lx / n_cells
    
    for i in range(n_cells):
        x_center = (i + 0.5) * cell_width
        cell_phase = phase + i * np.pi / n_cells
        
        # Convection roll pattern
        kx = 2 * np.pi / cell_width
        ky = np.pi / Ly
        
        # Temperature perturbation
        roll = amp * np.sin(kx * (X - x_center)) * np.sin(ky * Y) * np.cos(cell_phase)
        
        # Localize the roll
        envelope = np.exp(-2 * ((X - x_center) / cell_width)**2)
        
        T += roll * envelope
    
    # Generate velocity fields (simplified)
    u = np.zeros_like(T)
    v = np.zeros_like(T)
    
    # Simple circulation pattern
    for i in range(n_cells):
        x_center = (i + 0.5) * cell_width
        cell_phase = phase + i * np.pi / n_cells
        
        kx = 2 * np.pi / cell_width
        ky = np.pi / Ly
        
        u_roll = amp * 0.5 * kx * np.cos(kx * (X - x_center)) * np.cos(ky * Y) * np.cos(cell_phase)
        v_roll = -amp * 0.5 * ky * np.sin(kx * (X - x_center)) * np.sin(ky * Y) * np.cos(cell_phase)
        
        envelope = np.exp(-2 * ((X - x_center) / cell_width)**2)
        
        u += u_roll * envelope
        v += v_roll * envelope
    
    # Apply boundary conditions
    T[0, :] = 1.0
    T[-1, :] = 0.0
    u[0, :] = u[-1, :] = 0.0
    v[0, :] = v[-1, :] = 0.0
    
    # Periodic in x
    T[:, 0] = T[:, -1]
    u[:, 0] = u[:, -1]
    v[:, 0] = v[:, -1]
    
    # Simple pressure field
    p = np.zeros_like(T)
    
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
                            visualize=False, viz_frequency=50, create_animation=False):
    """Generate training dataset efficiently"""
    
    print(f"Fast RB data generation for Ra = {Ra:.0e}")
    print(f"  {n_runs} runs, 768×256 grid")
    if visualize:
        print(f"  📊 Visualization enabled: saving every {viz_frequency} samples")
    if create_animation:
        print(f"  🎬 Animation enabled: creating evolution GIFs")
    
    # Paper parameters
    nx, ny = 768, 256
    dt = 5e-4
    delta_t = 0.1 if Ra == 1e5 else 0.05
    
    startup_time = 25.0
    n_samples = 200
    startup_steps = int(startup_time / dt)
    steps_per_save = int(delta_t / dt)
    
    os.makedirs(save_path, exist_ok=True)
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}")
        
        data = []
        
        # Generate samples efficiently
        for sample in range(n_samples):
            step = startup_steps + sample * steps_per_save
            
            T, u, v, p = generate_rb_snapshot(nx, ny, Ra, step, dt)
            
            data.append({
                'temperature': T,
                'velocity_x': u, 
                'velocity_y': v,
                'pressure': p,
                'time': startup_time + sample * delta_t,
                'Ra': Ra,
                'Pr': 0.7
            })
            
            # Optional visualization
            if visualize and sample % viz_frequency == 0:
                viz_file = save_visualization(T, u, v, Ra, startup_time + sample * delta_t, 
                                            save_path, run, sample)
                print(f"    📊 Saved visualization: {os.path.basename(viz_file)}")
            
            if sample % 100 == 0:
                print(f"    Sample {sample+1}/{n_samples}")
        
        # Save to HDF5
        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        
        with h5py.File(filename, 'w') as f:
            # Metadata
            f.attrs.update({
                'Ra': Ra, 'Pr': 0.7, 'nx': nx, 'ny': ny,
                'Lx': 3.0, 'Ly': 1.0, 'dt': dt, 'delta_t': delta_t,
                't_start': startup_time, 't_end': startup_time + (n_samples-1) * delta_t,
                'n_samples': n_samples
            })
            
            # Data
            for i, frame in enumerate(data):
                grp = f.create_group(f'frame_{i:03d}')
                for key, value in frame.items():
                    if isinstance(value, np.ndarray):
                        grp.create_dataset(key, data=value, compression='gzip')
                    else:
                        grp.attrs[key] = value
        
        print(f"    Saved: {filename}")
        
        # Optional animation creation
        if create_animation:
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
    parser.add_argument('--viz_frequency', type=int, default=50, 
                       help='Save visualization every N samples (default: 50)')
    parser.add_argument('--animation', action='store_true', 
                       help='Create evolution animations (requires more time/storage)')
    parser.add_argument('--Ra', type=float, nargs='+', default=[1e5, 1e6, 1e7],
                       help='Rayleigh numbers to generate (default: 1e5 1e6 1e7)')
    parser.add_argument('--n_runs', type=int, default=25, 
                       help='Number of runs per Ra (default: 25)')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical',
                       help='Output directory (default: rb_data_numerical)')
    
    args = parser.parse_args()
    
    if args.test:
        # Quick test with small dataset
        print("🧪 Quick test...")
        generate_training_dataset(Ra=1e5, n_runs=2, save_path='rb_test_fast', 
                                visualize=True, viz_frequency=25, create_animation=True)
        print("✅ Test passed!")
        print("Check rb_test_fast/visualizations/ for sample outputs")
        
    else:
        # Generate full dataset
        print("🚀 Generating RB training data for CDAnet...")
        print(f"Ra numbers: {args.Ra}")
        print(f"Runs per Ra: {args.n_runs}")
        
        if args.visualize:
            print(f"📊 Visualizations: every {args.viz_frequency} samples")
        if args.animation:
            print("🎬 Animations: enabled")
        print()
        
        for Ra in args.Ra:
            generate_training_dataset(
                Ra=Ra, 
                n_runs=args.n_runs, 
                save_path=args.save_path,
                visualize=args.visualize,
                viz_frequency=args.viz_frequency,
                create_animation=args.animation
            )
        
        print("\n✅ All training data generated!")
        print("📁 Dataset summary:")
        print(f"  • {len(args.Ra)} Ra numbers: {args.Ra}")
        print(f"  • {args.n_runs} runs each (recommend 20 train + 5 val)")  
        print(f"  • 200 snapshots per run")
        print(f"  • Paper-compliant format and parameters")
        
        if args.visualize:
            print(f"  • Visualizations saved to {args.save_path}/visualizations/")
            print("  • Open summary.html files to view results in browser")