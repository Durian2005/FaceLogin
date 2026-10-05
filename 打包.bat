@echo off
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
set "BUILDER=%~dp0tools\build_exe.py"

if not exist "%PY%" (
  echo [ERROR] Python venv not found:
  echo   %PY%
  echo Run the windowed launcher once to create it.
  pause
  exit /b 1
)

if not exist "%BUILDER%" (
  echo [ERROR] Build script not found:
  echo   %BUILDER%
  pause
  exit /b 1
)

echo ==============================================================
echo   Building FaceLogin.exe  (no-install folder)
echo   This usually takes 1-3 minutes. Please wait.
echo ==============================================================
echo.

"%PY%" "%BUILDER%" %*

if errorlevel 1 (
  echo.
  echo [ERROR] Build failed. See the messages above.
  pause
  exit /b 1
)

echo.
echo ==============================================================
echo   Verifying the packaged build  (real HTTP probes)
echo ==============================================================
echo.

"%PY%" "%~dp0tools\smoke_frozen.py"

if errorlevel 1 (
  echo.
  echo [ERROR] Packaged build FAILED verification.
  echo         The package is not trustworthy - do not distribute it.
  echo         See the probe results above.
  pause
  exit /b 1
)

echo.
echo Output folder: dist\FaceLogin
echo Double-click FaceLogin.exe there to run.
pause
exit /b 0
