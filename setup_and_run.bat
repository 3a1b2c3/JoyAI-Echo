@echo off
setlocal enabledelayedexpansion
REM JoyAI-Echo: Download models, setup environment, run examples

cd /d "%~dp0"

echo.
echo ================================================================================
echo JoyAI-Echo - Setup ^& Run
echo ================================================================================
echo.

REM 1. Check Python
echo [1/5] Checking Python...

set "PYEXE="
for %%P in (python.exe) do (
    if not defined PYEXE (
        where %%P >nul 2>&1 && set "PYEXE=%%P"
    )
)

if not defined PYEXE (
    echo ERROR: Python not found. Install Python 3.11 from https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2" %%V in ('%PYEXE% --version 2^>^&1') do set "PYVER=%%V"
echo   Python: %PYEXE% ^(%PYVER%^)

REM 2. Create venv
echo.
echo [2/5] Setting up virtual environment...
if not exist ".venv" (
    %PYEXE% -m venv .venv
    echo   Created .venv
) else (
    echo   .venv already exists
)

call .venv\Scripts\activate.bat
echo   Activated .venv

REM 3. Install dependencies
echo.
echo [3/5] Installing dependencies (CUDA 12.8)...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1

echo   Installing PyTorch...
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo   Installing requirements...
python -m pip install -r requirements.txt

for %%D in (ltx-core ltx-pipelines ltx-distillation) do (
    if exist "%%D" (
        echo   Installing %%D...
        python -m pip install -e "%%D"
    )
)

echo   Dependencies installed

REM 4. Download models
echo.
echo [4/5] Downloading models (first run only)...
echo.
call download_models.bat
if errorlevel 1 exit /b 1

REM 5. Run examples
echo.
echo [5/5] Ready to run examples!
echo.
echo ================================================================================
echo QUICKSTART
echo ================================================================================
echo.
echo Run all prompt files in prompts\ (each is a multi-shot story):
echo   python inference.py
echo.
echo Run a single prompt file:
echo   python inference.py --prompts-glob "test_001.json"
echo.
echo Custom resolution:
echo   python inference.py --video-width 1024 --video-height 576
echo.
echo ================================================================================
echo.

set /p RUNNOW="Run one example prompt file now (test_001.json, ~15 shots, ~50-60 min on A100/H100)? (y/n) "
if /i "%RUNNOW%"=="y" (
    echo.
    echo Running inference on prompts\test_001.json...
    echo Expected output size: ~2-3 GB
    echo.
    python inference.py --prompts-glob "test_001.json"
    if errorlevel 1 (
        echo ERROR: Inference failed
        exit /b 1
    )
    echo.
    echo Complete!
    echo.
    echo Outputs:
    for /r inference_result %%F in (combined_shots.mp4) do (
        if exist "%%F" echo   %%F
    )
) else (
    echo.
    echo Setup complete!
    echo.
    echo Next steps:
    echo   1. Run all prompt files: python inference.py
    echo   2. Run one prompt file:  python inference.py --prompts-glob "test_001.json"
    echo   3. Custom resolution:    python inference.py --video-width 1024 --video-height 576
    echo   Or use the interactive picker: run_examples.bat
)

echo.
endlocal
