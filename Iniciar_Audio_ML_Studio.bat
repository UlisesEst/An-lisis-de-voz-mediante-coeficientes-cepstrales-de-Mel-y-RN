@echo off
title Audio ML Studio
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Iniciando Audio ML Studio...
    ".venv\Scripts\python.exe" desktop_app.py
) else (
    echo No se encontro el entorno virtual .venv. Intentando con Python global...
    python desktop_app.py
)
