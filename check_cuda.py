#!/usr/bin/env python3
"""
Quick CUDA check script for Ubuntu systems.
"""

import torch

def check_cuda():
    print("🔍 CUDA System Check")
    print("=" * 40)

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get CUDA details
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # GPU details
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # Test GPU computation
        print("\n🧪 Testing GPU computation...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.t())
            print("✅ GPU computation test passed!")

            # Clean up
            del x, y
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    else:
        print("❌ CUDA not available. Please install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    print("=" * 40)
    return cuda_available

if __name__ == "__main__":
    check_cuda()