import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check build info (newer TensorFlow versions may not have these keys)
build_info = tf.sysconfig.get_build_info()
print("Available build info keys:", list(build_info.keys()))

# Try to get CUDA info if available
if 'cuda_version' in build_info:
    print("CUDA version:", build_info['cuda_version'])
else:
    print("CUDA version info not available in build_info")

if 'cudnn_version' in build_info:
    print("cuDNN version:", build_info['cudnn_version'])
else:
    print("cuDNN version info not available in build_info")

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
print("GPU devices:", gpus)
print("Number of GPUs:", len(gpus))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Check GPU memory and details
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"GPU {i} details:", gpu_details)
        except:
            print(f"Could not get details for GPU {i}")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("GPU computation successful!")
            print("Result:", c)
    except Exception as e:
        print(f"GPU computation failed: {e}")
else:
    print("No GPU detected")

# Additional GPU information
print(f"\nTensorFlow built with GPU support: {tf.test.is_built_with_cuda()}")
print(f"TensorFlow can access GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")