@echo off
echo ===================================================
echo       🛡️ Starting FraudGuard AI System 🛡️
echo ===================================================
echo.
echo Initializing Python Environment...
echo.

:: Check if Streamlit is installed
python -c "import streamlit" 2>NUL
if %errorlevel% neq 0 (
    echo Error: Streamlit is not installed.
    echo Installing requirements...
    pip install -r requirements.txt
)

echo Launching Application...
python -m streamlit run app.py

pause
