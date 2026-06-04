@echo off
setlocal

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_dashboard.ps1" %*

if errorlevel 1 (
  echo.
  pause
)

