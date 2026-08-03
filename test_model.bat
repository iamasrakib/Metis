@echo off
rem ============================================================
rem  test_model.bat - Metis model tester launcher
rem
rem  Runs test_model.py with the Python that actually has torch
rem  installed (the repo's .venv is currently empty, so we pin
rem  the global Python 3.11 which has torch + metis).
rem
rem  Usage:
rem    test_model.bat info
rem    test_model.bat generate --prompt "Why do cats purr?"
rem    test_model.bat chat
rem    test_model.bat test
rem ============================================================
setlocal
set "ROOT=%~dp0"

rem 1) Known-good Python (torch + metis installed here)
set "PY=C:\Users\iamas\AppData\Local\Programs\Python\Python311\python.exe"
if exist "%PY%" goto :run

rem 2) Fall back to whatever python is on PATH
set "PY=python"

:run
"%PY%" "%ROOT%test_model.py" %*
exit /b %errorlevel%
