@echo off
REM OpenClip Pro - Ngrok Deployment Script for Windows
REM This script automates the deployment process on Windows

echo ============================================
echo OpenClip Pro - Ngrok Deployment for Windows
echo ============================================
echo.

REM Check Python - try both python and py commands
set PYTHON_CMD=
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    echo Found Python using 'python' command
) else (
    py -3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py -3
        echo Found Python using 'py -3' command
    ) else (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.8+ from https://python.org
        echo.
        echo If Python is installed, try one of these:
        echo 1. Add Python to your PATH environment variable
        echo 2. Use 'py -3 deploy_ngrok.py' directly
        pause
        exit /b 1
    )
)

REM Check if virtual environment exists
if exist "venv" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using global Python.
)

REM Check for ffmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: ffmpeg is not installed or not in PATH
    echo Please install ffmpeg and add it to your PATH
    echo Download from: https://ffmpeg.org/download.html
    pause
    exit /b 1
)

REM Check for ngrok
ngrok version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: ngrok is not installed or not in PATH
    echo Please download ngrok from https://ngrok.com/download
    echo And add it to your system PATH
    pause
    exit /b 1
)

REM Check for required Python packages
echo Checking Python dependencies...
%PYTHON_CMD% -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Create necessary directories
if not exist ".streamlit" mkdir .streamlit
if not exist "tmp" mkdir tmp

REM Menu for deployment options
:menu
echo.
echo Select deployment option:
echo 1. Quick deployment (no authentication)
echo 2. Deployment with ngrok auth token
echo 3. Custom deployment (advanced)
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto quick_deploy
if "%choice%"=="2" goto auth_deploy
if "%choice%"=="3" goto custom_deploy
if "%choice%"=="4" goto end
echo Invalid choice. Please try again.
goto menu

:quick_deploy
echo.
echo Starting quick deployment...
echo.
%PYTHON_CMD% deploy_ngrok.py
goto end

:auth_deploy
echo.
set /p token="Enter your ngrok auth token: "
echo.
echo Starting authenticated deployment...
echo.
%PYTHON_CMD% deploy_ngrok.py --auth-token %token%
goto end

:custom_deploy
echo.
set /p port="Enter port number (default 8501): "
if "%port%"=="" set port=8501
set /p token="Enter ngrok auth token (optional, press Enter to skip): "

echo.
echo Starting custom deployment on port %port%...
echo.

if "%token%"=="" (
    %PYTHON_CMD% deploy_ngrok.py --port %port%
) else (
    %PYTHON_CMD% deploy_ngrok.py --port %port% --auth-token %token%
)
goto end

:end
echo.
echo Deployment process completed.
pause 