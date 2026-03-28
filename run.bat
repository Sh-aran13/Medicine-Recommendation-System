@echo off
setlocal

REM Move to project root (location of this script)
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
  echo.
  echo Run these commands once from project root:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate.bat
  echo   pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  pause
  exit /b 1
)

if not exist "mrs_web\manage.py" (
  echo [ERROR] Could not find mrs_web\manage.py
  pause
  exit /b 1
)

cd /d "%~dp0mrs_web"
echo Starting Django development server at http://127.0.0.1:8000/
echo Press Ctrl+C to stop.
echo.
python manage.py runserver

set EXIT_CODE=%ERRORLEVEL%
echo.
echo Server exited with code %EXIT_CODE%.
pause
endlocal & exit /b %EXIT_CODE%
