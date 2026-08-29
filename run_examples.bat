@echo off
setlocal enabledelayedexpansion
REM JoyAI-Echo: Interactive example runner

cd /d "%~dp0"

echo.
echo ================================================================================
echo JoyAI-Echo - Run Examples
echo ================================================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found. Run setup_and_run.bat first
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Activated .venv
echo.

dir /b "checkpoints\echo15_*" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Models not downloaded. Run download_models.bat first
    exit /b 1
)

set COUNT=0
for %%F in (prompts\*.json) do (
    set /a COUNT+=1
    set "PROMPT_!COUNT!=%%~nxF"
)

echo Available examples:
echo.
echo   1. Run all prompt files in prompts\ (!COUNT! found, each is a multi-shot story)
echo   2. Run a single prompt file
echo   3. Custom resolution (e.g. 1024x576)
echo   4. Custom FPS (e.g. 30)
echo.

set /p CHOICE="Choose example (1-4): "

if "%CHOICE%"=="1" (
    echo.
    echo Running all prompt files...
    python inference.py
) else if "%CHOICE%"=="2" (
    echo.
    echo Prompt files:
    for /l %%i in (1,1,!COUNT!) do echo   %%i. !PROMPT_%%i!
    set /p FCHOICE="Choose file (1-!COUNT!): "
    call set "SELECTED=%%PROMPT_!FCHOICE!%%"
    echo.
    echo Running !SELECTED!...
    python inference.py --prompts-glob "!SELECTED!"
) else if "%CHOICE%"=="3" (
    echo.
    set /p WIDTH="Width: "
    set /p HEIGHT="Height: "
    echo Running with custom resolution (!WIDTH!x!HEIGHT!)...
    python inference.py --video-width !WIDTH! --video-height !HEIGHT!
) else if "%CHOICE%"=="4" (
    echo.
    set /p FPS="FPS: "
    echo Running with custom FPS (!FPS!)...
    python inference.py --video-fps !FPS!
) else (
    echo Invalid choice
    exit /b 1
)

echo.
echo ================================================================================
echo Complete!
echo ================================================================================
echo.

echo Results:
for /r inference_result %%F in (combined_shots.mp4) do (
    if exist "%%F" echo   %%F
)

echo.
echo View a video with: start ^<path-from-above^>
echo.
endlocal
