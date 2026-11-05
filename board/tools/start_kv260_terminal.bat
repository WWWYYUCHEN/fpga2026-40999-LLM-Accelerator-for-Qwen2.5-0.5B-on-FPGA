@echo off
setlocal
chcp 65001 >nul
mode con: cols=120 lines=40
title KV260 Qwen2.5-0.5B Interactive UART
set "KV260_COM=COM11"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_kv260_terminal.ps1"
echo.
echo KV260 terminal closed.
pause
