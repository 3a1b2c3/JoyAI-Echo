@echo off
setlocal enabledelayedexpansion
REM JoyAI-Echo: Download models (Echo 1.5 checkpoints)
REM Variant override: set VARIANT=echo15_full_dmd (BF16, 46.14 GB) or echo15_fp4 (FP4, 22.81 GB)
REM Default: echo15_fp8 (FP8, 27.62 GB)

cd /d "%~dp0"

if not defined VARIANT set "VARIANT=echo15_fp8"

echo.
echo ================================================================================
echo JoyAI-Echo - Download Models (variant: %VARIANT%)
echo ================================================================================
echo.

set "CKPT_DIR=%CD%\checkpoints"
if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"

echo Checkpoints: %CKPT_DIR%
echo.
echo Files to download:
echo   1. %VARIANT% (Echo 1.5 checkpoint)
echo   2. gemma-3-12b-it (text encoder)       ~24 GB
echo.

python -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo Installing huggingface_hub...
    python -m pip install --quiet -U huggingface_hub
)

REM ============================================================================
REM 1. Echo 1.5 checkpoint variant from jdopensource/JoyAI-Echo
REM ============================================================================
echo [1/2] Echo Model (%VARIANT%)
echo.

set "VARIANT_DIR=%CKPT_DIR%\%VARIANT%"

if exist "%VARIANT_DIR%\*" (
    echo Already exists: %VARIANT_DIR%
) else (
    echo Downloading %VARIANT% ^(this will take a while^)...
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jdopensource/JoyAI-Echo', allow_patterns=['%VARIANT%/*'], local_dir=r'%CKPT_DIR%')"
    if errorlevel 1 (
        echo Download failed. URL: https://huggingface.co/jdopensource/JoyAI-Echo/tree/main/%VARIANT%
        exit /b 1
    )
    echo Downloaded: %VARIANT_DIR%
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
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/gemma-3-12b-it', local_dir=r'%GEMMA_DIR%')"
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
echo NOTE: this checkout's inference.py predates the Echo 1.5 checkpoint format
echo (checkpoint.json + %VARIANT%\*.safetensors^). It will not load these files
echo until the code is updated to match jd-opensource/JoyAI-Echo's echo_longvideo/ branch.
echo.
endlocal
