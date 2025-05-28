import os
import sys
import subprocess
import platform
import tensorflow as tf
import ctypes
import urllib.request
import zipfile
import shutil
from pathlib import Path

def check_gpu_status():
    """Check if TensorFlow can access the GPU."""
    print("TensorFlow version:", tf.__version__)
    print("Checking GPU availability...")
    
    # Check if CUDA is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPU(s):")
        for device in physical_devices:
            print(f"  - {device}")
        
        # Try to enable memory growth to avoid allocating all GPU memory
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Memory growth enabled for all GPUs")
        except Exception as e:
            print(f"Error enabling memory growth: {e}")
    else:
        print("No GPU detected by TensorFlow.")
    
    # Check if CUDA is visible
    print("\nChecking CUDA environment...")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Print relevant paths
    print("\nChecking system paths...")
    path = os.environ.get('PATH', '').split(os.pathsep)
    cuda_paths = [p for p in path if 'cuda' in p.lower()]
    if cuda_paths:
        print("CUDA paths found in PATH:")
        for p in cuda_paths:
            print(f"  - {p}")
    else:
        print("No CUDA paths found in PATH environment variable")
    
    # Check if CUDA DLLs are accessible
    print("\nChecking for CUDA DLLs...")
    cuda_dlls = ['cublas64_11.dll', 'cudart64_11.dll', 'cufft64_11.dll', 
                'curand64_11.dll', 'cusolver64_11.dll', 'cudnn64_8.dll']
    
    found_dlls = []
    for dll in cuda_dlls:
        try:
            handle = ctypes.WinDLL(dll)
            found_dlls.append(dll)
            del handle
        except Exception:
            pass
    
    if found_dlls:
        print("Found the following CUDA DLLs:")
        for dll in found_dlls:
            print(f"  - {dll}")
    else:
        print("No CUDA DLLs found in system path")

def check_nvidia_gpu():
    """Check if NVIDIA GPU is present and get details."""
    print("\nChecking for NVIDIA GPU...")
    try:
        # Try using nvidia-smi
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU detected using nvidia-smi:")
            # Extract relevant parts
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if i < 10:  # Just show the first few lines
                    print(line)
                if "Driver Version" in line:
                    print(line)
        else:
            print("nvidia-smi failed. GPU might not be present or driver not installed.")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi command not found. GPU drivers might not be installed.")

def check_direct_dll_presence():
    """Check if the required DLLs exist in common locations."""
    print("\nChecking common locations for CUDA/cuDNN DLLs:")
    possible_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\Program Files\\NVIDIA\\CUDNN"
    ]
    
    # Check CUDA versions
    cuda_versions = []
    for base_path in possible_paths:
        if os.path.exists(base_path):
            dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            for d in dirs:
                if d.startswith('v'):
                    cuda_versions.append(os.path.join(base_path, d))
    
    if cuda_versions:
        print("Found potential CUDA/cuDNN installations:")
        for path in cuda_versions:
            print(f"  - {path}")
            # Check for bin directory
            bin_dir = os.path.join(path, 'bin')
            if os.path.exists(bin_dir):
                dlls = [f for f in os.listdir(bin_dir) if f.endswith('.dll')]
                if dlls:
                    print(f"    Contains {len(dlls)} DLLs including:")
                    cuda_specific = [dll for dll in dlls if 'cu' in dll.lower() or 'cuda' in dll.lower()]
                    for dll in cuda_specific[:5]:  # Show only first 5
                        print(f"      - {dll}")
                    if len(cuda_specific) > 5:
                        print(f"      - ... and {len(cuda_specific)-5} more")
    else:
        print("No CUDA installations found in common locations.")

def fix_gpu_setup():
    """Try to help fix GPU setup for TensorFlow."""
    print("\n" + "="*50)
    print("GPU SETUP HELPER")
    print("="*50)
    
    print("\nBased on the diagnostics, here are some steps to fix your GPU setup:")
    
    print("\n1. Install appropriate NVIDIA drivers")
    print("   - Download latest drivers from: https://www.nvidia.com/Download/index.aspx")
    print("   - Make sure the driver is compatible with your RTX 3070")
    
    print("\n2. Install CUDA Toolkit")
    print("   - For TensorFlow 2.10-2.12: CUDA 11.2")
    print("   - For TensorFlow 2.13+: CUDA 11.8 or CUDA 12.x")
    print("   - Download link: https://developer.nvidia.com/cuda-downloads")
    
    print("\n3. Install cuDNN")
    print("   - Download compatible cuDNN from: https://developer.nvidia.com/cudnn")
    print("   - Extract to your CUDA installation directory")
    
    print("\n4. Set up environment variables:")
    print("   - Add the following paths to your PATH environment variable:")
    print("     * C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\bin")
    print("     * C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\libnvvp")
    print("     * C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\extras\\CUPTI\\lib64")
    print("     * C:\\path\\to\\cudnn\\bin")
    
    print("\n5. Consider using conda environment with GPU packages pre-configured:")
    print("   conda create -n tf-gpu tensorflow-gpu")
    print("   conda activate tf-gpu")
    
    print("\n6. For Windows 11, you might need to check if your GPU is set as high-performance in:")
    print("   Settings > System > Display > Graphics > Default Graphics Settings")
    
    print("\nWould you like to attempt an automatic fix by downloading a compatible")
    print("version of TensorFlow that matches your installed CUDA version? (y/n)")
    
    # In a real script, we would wait for input here

def main():
    """Main function to run diagnostics."""
    print("="*50)
    print(f"TensorFlow GPU Setup Diagnostics on {platform.system()} {platform.release()}")
    print("="*50)
    
    check_gpu_status()
    check_nvidia_gpu()
    check_direct_dll_presence()
    fix_gpu_setup()
    
    print("\n" + "="*50)
    print("RECOMMENDATION")
    print("="*50)
    print("Based on these diagnostics, consider creating a new conda environment")
    print("with compatible CUDA/cuDNN and TensorFlow versions:")
    print("\n1. If you don't have conda, install Miniconda from:")
    print("   https://docs.conda.io/en/latest/miniconda.html")
    print("\n2. Open Anaconda prompt and run these commands:")
    print("   conda create -n asl-gpu python=3.10")
    print("   conda activate asl-gpu")
    print("   conda install tensorflow=2.10 cudatoolkit=11.2 cudnn=8.1 -c conda-forge")
    print("\n3. Then run your script using this new environment:")
    print("   conda activate asl-gpu")
    print("   python train.py")

if __name__ == "__main__":
    main()