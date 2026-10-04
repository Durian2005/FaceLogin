@echo off
setlocal
cd /d "%~dp0"
set "OUT=%~dp0diagnose_result.txt"
echo === Face Login Diagnose === > "%OUT%"
echo. >> "%OUT%"

echo [1] Folder >> "%OUT%"
echo CWD: %CD% >> "%OUT%"
echo. >> "%OUT%"

echo [2] Python venv >> "%OUT%"
if exist "%~dp0.venv\Scripts\python.exe" (
  echo OK: venv python exists >> "%OUT%"
  "%~dp0.venv\Scripts\python.exe" --version >> "%OUT%" 2>&1
) else (
  echo MISSING: .venv\Scripts\python.exe >> "%OUT%"
)
echo. >> "%OUT%"

echo [3] app files >> "%OUT%"
if exist "%~dp0app\run.py" (echo OK: app\run.py >> "%OUT%") else (echo MISSING: app\run.py >> "%OUT%")
if exist "%~dp0app\server.py" (echo OK: app\server.py >> "%OUT%") else (echo MISSING: app\server.py >> "%OUT%")
echo. >> "%OUT%"

echo [4] models >> "%OUT%"
if exist "%~dp0models\face_detection_yunet_2023mar.onnx" (echo OK: yunet >> "%OUT%") else (echo MISSING: yunet >> "%OUT%")
if exist "%~dp0models\face_recognition_sface_2021dec.onnx" (echo OK: sface >> "%OUT%") else (echo MISSING: sface >> "%OUT%")
echo. >> "%OUT%"

echo [5] import test >> "%OUT%"
"%~dp0.venv\Scripts\python.exe" -c "import flask,numpy,cv2;print('flask',flask.__version__);print('numpy',numpy.__version__);print('cv2',cv2.__version__)" >> "%OUT%" 2>&1
echo. >> "%OUT%"

echo [6] port 5000 >> "%OUT%"
netstat -ano | findstr "127.0.0.1:5000" >> "%OUT%" 2>&1
echo. >> "%OUT%"

echo [7] camera quick test >> "%OUT%"
"%~dp0.venv\Scripts\python.exe" -c "import cv2;c=cv2.VideoCapture(0,cv2.CAP_DSHOW);print('opened',c.isOpened());ok,f=c.read() if c.isOpened() else (False,None);print('read',ok);c.release()" >> "%OUT%" 2>&1
echo. >> "%OUT%"

echo === done === >> "%OUT%"
echo Diagnose finished. Result saved to:
echo %OUT%
echo.
pause
