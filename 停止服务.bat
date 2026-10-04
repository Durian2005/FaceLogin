@echo off
setlocal
echo Stopping face login service...

set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:5000" ^| findstr "LISTENING"') do (
  echo Killing PID %%a
  taskkill /F /PID %%a >nul 2>nul
  set "FOUND=1"
)

if "%FOUND%"=="0" (
  echo No running service found.
) else (
  echo Service stopped.
)

timeout /t 2 /nobreak >nul
