@echo off
setlocal
cd /d "%~dp0"

set "LOG=%~dp0launch_log.txt"
echo [%date% %time%] dispatch start > "%LOG%"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "RUNNER=%~dp0app\run.py"
set "FACELOGIN_MODELS=%~dp0models"

if not exist "%VENV_PY%" (
  echo [ERROR] venv python missing: %VENV_PY% >> "%LOG%"
  echo [ERROR] Run the windowed .bat launcher once first to create the venv. >> "%LOG%"
  type "%LOG%"
  pause
  exit /b 1
)
if not exist "%RUNNER%" (
  echo [ERROR] runner missing: %RUNNER% >> "%LOG%"
  type "%LOG%"
  pause
  exit /b 1
)
if not exist "%FACELOGIN_MODELS%\face_detection_yunet_2023mar.onnx" (
  echo [ERROR] model not found: %FACELOGIN_MODELS%\face_detection_yunet_2023mar.onnx >> "%LOG%"
  type "%LOG%"
  pause
  exit /b 1
)

echo [INFO] starting minimized service window >> "%LOG%"
start "FaceLoginService" /MIN "%VENV_PY%" "%RUNNER%"
echo.
echo Service window launched (minimized).
echo To stop it: double-click the stop .bat, or close that minimized window.
timeout /t 3 /nobreak >nul
