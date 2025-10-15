@echo off
echo ======================================================
echo Fixing NumPy version compatibility with TensorFlow
echo ======================================================

REM Check if Conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda is not installed or not in PATH.
    echo Please run this script from an Anaconda Prompt.
    pause
    exit /b
)

echo Activating asl-gpu environment...
call conda activate asl-gpu

REM Uninstall current numpy
echo Removing current NumPy installation...
call pip uninstall -y numpy

REM Install compatible numpy version
echo Installing compatible NumPy version...
call pip install numpy==1.24.3

REM Verify TensorFlow works with this numpy
echo Testing TensorFlow import...
python -c "import tensorflow as tf; print('TensorFlow import successful!'); print('TensorFlow version:', tf.__version__); print('NumPy version:', tf.numpy.__version__)" 

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo There was an error importing TensorFlow.
    echo Trying alternative approach with specific package versions...
    
    REM Alternative approach with specific package versions
    call pip uninstall -y tensorflow tensorflow-estimator keras h5py
    call pip install h5py==3.7.0
    call pip install tensorflow==2.10.0
)

echo.
echo ======================================================
echo Setup complete! 
echo.
echo Run your script with:
echo   python train.py
echo ======================================================

pause