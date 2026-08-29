@echo off
setlocal
cd /d "%~dp0"

set "BASH_EXE=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH_EXE%" set "BASH_EXE=%LocalAppData%\Programs\Git\bin\bash.exe"

if not exist "%BASH_EXE%" (
  echo Git Bash is required. Install Git for Windows, then run this command again.
  exit /b 1
)

"%BASH_EXE%" start_local.sh
exit /b %ERRORLEVEL%
