@echo off
cls
echo ===============================================
echo     Ultra Trading Bot - Professional Edition
echo               Version 2.0
echo ===============================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Setting up virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo Please ensure Python 3.8-3.11 is installed
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check Python version
echo [INFO] Checking Python version...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found
    echo Please install Python 3.8-3.11
    pause
    exit /b 1
)

REM Install/update requirements
echo [INFO] Installing/updating requirements...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Check MetaTrader5 installation
echo [INFO] Checking MetaTrader5 installation...
python -c "import MetaTrader5; print('MetaTrader5 OK')" 2>nul
if errorlevel 1 (
    echo [WARNING] MetaTrader5 library not found
    echo This is normal for Linux/Mac systems
    echo The bot will run in simulation mode
) else (
    echo [SUCCESS] MetaTrader5 library found
)

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "backups" mkdir backups

REM Start the bot
echo.
echo ===============================================
echo            Starting Ultra Trading Bot
echo ===============================================
echo.
echo [INFO] Bot starting on: http://localhost:8000
echo [INFO] Press Ctrl+C to stop the bot
echo.

REM Run the bot with error handling
python advanced_server.py
if errorlevel 1 (
    echo.
    echo [ERROR] Bot crashed or stopped unexpectedly
    echo Check the logs for more information
    pause
    exit /b 1
)

echo.
echo [INFO] Bot stopped successfully
pause