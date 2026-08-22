@echo off
REM Truth Scan - always launches with the correct project virtualenv,
REM so the NLI model and every dependency are available.
cd /d "%~dp0"
echo Starting AI-Powered Fake News Detection (venv)...
"venv\Scripts\python.exe" -m streamlit run streamlit_app\app.py
pause
