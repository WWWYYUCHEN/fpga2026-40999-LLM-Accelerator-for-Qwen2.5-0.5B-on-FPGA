@echo off
setlocal
chcp 65001 >nul
title KV260 SD Verification
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_kv260_sd.ps1"
echo.
pause
