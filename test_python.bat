@echo off
echo ============================================
echo Testing Python Installation
echo ============================================
echo.

echo Checking for 'python' command:
python --version 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Python found via 'python' command
) else (
    echo [NOT FOUND] 'python' command not available
)

echo.
echo Checking for 'py' command (Windows Python Launcher):
py -3 --version 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Python found via 'py -3' command
) else (
    echo [NOT FOUND] 'py' command not available
)

echo.
echo ============================================
echo.
echo If Python was found, you can now run:
echo - deploy_ngrok.bat (for automated deployment)
echo - py -3 check_deployment.py (to check environment)
echo - py -3 deploy_ngrok.py (for manual deployment)
echo.
pause 