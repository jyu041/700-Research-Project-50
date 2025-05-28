@echo off
echo Setting up TensorFlow GPU environment for ASL Classification
echo =====================================================

REM Check if Conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda is not installed or not in PATH.
    echo Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    echo and run this script again.
    pause
    exit /b
)

echo Creating new Conda environment with TensorFlow GPU support...
call conda create -n asl-gpu python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create Conda environment.
    pause
    exit /b
)

echo Activating environment and installing packages...
call conda activate asl-gpu
call conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
call pip install tensorflow==2.10.0

echo Installing other required packages...
call pip install opencv-python matplotlib seaborn pandas scikit-learn

echo ==============================================================
echo Setup complete! 
echo.
echo To use this environment:
echo 1. Open Anaconda Prompt
echo 2. Run: conda activate asl-gpu
echo 3. Run: python train.py
echo ==============================================================

pause