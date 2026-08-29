@echo off
setlocal enabledelayedexpansion
REM JoyAI-Echo: Download models

cd /d "%~dp0"

echo.
echo ================================================================================
echo JoyAI-Echo - Download Models
echo ================================================================================
echo.

set "CKPT_DIR=%CD%\checkpoints"
if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"

echo Checkpoints: %CKPT_DIR%
echo.
echo Files to download:
echo   1. echo-longvideo-release.safetensors  ~46 GB
echo   2. gemma-3-12b-it (text encoder)       ~24 GB
echo   ------------------------------------------------
echo   Total:                                 ~70 GB
echo.

python -m pip show huggingface_hub >nul 2>&1
if errorlevel 1 (
    echo Installing huggingface_hub...
    python -m pip install --quiet -U "huggingface_hub[cli]"
)

REM ============================================================================
REM 1. Echo model from jdopensource/JoyAI-Echo
REM ============================================================================
echo [1/2] Echo Model
echo.

set "MODEL_FILE=%CKPT_DIR%\echo-longvideo-release.safetensors"

if exist "%MODEL_FILE%" (
    echo Already exists: %MODEL_FILE%
) else (
    echo Downloading ^(~46 GB, this will take a while^)...
    hf download jdopensource/JoyAI-Echo echo-longvideo-release.safetensors --local-dir "%CKPT_DIR%"
    if errorlevel 1 (
        echo Download failed. URL: https://huggingface.co/jdopensource/JoyAI-Echo
        exit /b 1
    )
    echo Downloaded: %MODEL_FILE%
)

REM ============================================================================
REM 2. Gemma text encoder (required - inference.py fails to start without it)
REM ============================================================================
echo.
echo [2/2] Text Encoder (Gemma 3 12B Instruct)
echo.

set "GEMMA_DIR=%CKPT_DIR%\gemma-3-12b"

if exist "%GEMMA_DIR%\*" (
    echo Already exists: %GEMMA_DIR%
) else (
    echo Downloading ^(~24 GB, this will take a while^)...
    hf download google/gemma-3-12b-it --local-dir "%GEMMA_DIR%"
    if errorlevel 1 (
        echo Download failed.
        echo Gemma is gated: accept the license at https://huggingface.co/google/gemma-3-12b-it
        echo then log in with: hf auth login
        exit /b 1
    )
    echo Downloaded: %GEMMA_DIR%
)

echo.
echo ================================================================================
echo Download Complete
echo ================================================================================
echo.

if exist "%MODEL_FILE%" (
    echo Ready to run examples
    echo.
    echo Next: run_examples.bat
) else (
    echo Main model missing
    exit /b 1
)

echo.
endlocal
