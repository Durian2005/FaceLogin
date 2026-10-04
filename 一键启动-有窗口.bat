@echo off
setlocal
cd /d "%~dp0"

set "LOG=%~dp0launch_log.txt"
echo [%date% %time%] starting > "%LOG%"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "RUNNER=%~dp0app\run.py"
set "FACELOGIN_MODELS=%~dp0models"

if not exist "%RUNNER%" (
  echo [ERROR] runner not found: %RUNNER% >> "%LOG%"
  echo [ERROR] expected file: %RUNNER% >> "%LOG%"
  type "%LOG%"
  pause
  exit /b 1
)

if not exist "%FACELOGIN_MODELS%\face_detection_yunet_2023mar.onnx" (
  echo [ERROR] model not found in %FACELOGIN_MODELS% >> "%LOG%"
  echo [ERROR] expected file: %FACELOGIN_MODELS%\face_detection_yunet_2023mar.onnx >> "%LOG%"
  type "%LOG%"
  pause
  exit /b 1
)

if not exist "%VENV_PY%" (
  echo [INFO] venv missing, creating... >> "%LOG%"
  py -3.13 -m venv "%~dp0.venv" >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] venv creation failed >> "%LOG%"
    type "%LOG%"
    pause
    exit /b 1
  )
  "%VENV_PY%" -m pip install flask==3.1.3 numpy==2.5.3 opencv-python-headless==5.0.0.93 >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] pip install failed >> "%LOG%"
    type "%LOG%"
    pause
    exit /b 1
  )
)

echo [INFO] launching service >> "%LOG%"
echo.
echo Service starting, browser will open automatically...
echo Keep this window open. Close it to stop the service.
echo.

"%VENV_PY%" "%RUNNER%" >> "%LOG%" 2>&1

echo.
echo Service stopped.
pause
