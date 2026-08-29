@echo off
setlocal
cd /d "%~dp0.."

where uv >nul 2>nul
if errorlevel 1 (
  echo [Echo Server] uv was not found. Install uv and reopen Command Prompt.
  exit /b 1
)

if not exist "configs\server.consumer.yaml" (
  echo [Echo Server] Missing configs\server.consumer.yaml
  exit /b 1
)

echo [Echo Server] Starting on the configured host and port...
uv run python server.py --config configs/server.consumer.yaml %*
