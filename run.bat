@echo off
setlocal
echo ================================================
echo  Product Insight AI - Launch System
echo ================================================
echo.

set VENV_PATH=.venv
set PYTHON_EXE=python

if exist "%VENV_PATH%\Scripts\python.exe" (
    echo [OK] Virtual environment (.venv) found.
    set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
) else (
    echo [!] Virtual environment not found. Checking global python...
)

echo [1/4] Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [2/4] Downloading NLTK data...
"%PYTHON_EXE%" -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

echo.
echo [3/4] Training ML model (if required)...
"%PYTHON_EXE%" train_model.py

echo.
echo [4/4] Starting Flask server...
echo  Open http://127.0.0.1:5000 in your browser
echo  Press Ctrl+C to stop
echo.
"%PYTHON_EXE%" app.py
pause
