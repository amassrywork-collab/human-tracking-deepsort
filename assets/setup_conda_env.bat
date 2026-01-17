@echo off
echo ============================================
echo   Setting up Deep Learning CV Environment
echo ============================================

REM Activate conda base (required)
call conda activate

REM Create environment if it does not exist
echo Creating conda environment (cv_dl_lab)...
conda create -n cv_dl_lab python=3.10 -y

REM Activate the environment
call conda activate cv_dl_lab

echo --------------------------------------------
echo Updating pip...
python -m pip install --upgrade pip

echo --------------------------------------------
echo Adding conda-forge channel...
conda config --add channels conda-forge
conda config --set channel_priority strict

echo --------------------------------------------
echo Installing scientific libraries via conda...
conda install scipy filterpy -y

echo --------------------------------------------
echo Installing Python packages via pip...
pip install ^
numpy>=1.24,<3.0 ^
opencv-python>=4.8,<5.0 ^
tqdm>=4.66,<5.0 ^
ultralytics>=8.2,<9.0 ^
deep-sort-realtime>=1.3.2,<2.0

echo --------------------------------------------
echo Testing installation...
python - << END
import numpy, cv2, torch, ultralytics
import deep_sort_realtime, scipy, filterpy, tqdm
print("Environment setup completed successfully!")
END

echo ============================================
echo   Environment is ready. You can start now.
echo ============================================

pause