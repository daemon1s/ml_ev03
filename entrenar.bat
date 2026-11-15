@echo off
setlocal
cd /d "%~dp0"
title Entrenamiento EV03
if "%~1"=="" (
    echo Uso: entrenar.bat nombre-de-la-corrida
    pause
    exit /b 1
)
if exist "%~dp0auditoria.env.bat" call "%~dp0auditoria.env.bat"
if not defined DATASET_DIR (
    echo Falta configurar DATASET_DIR. Copia auditoria.env.example.bat como auditoria.env.bat y completa los valores.
    pause
    exit /b 1
)
if not exist "%~dp0.venv-torch\Scripts\python.exe" (
    echo Falta el entorno .venv-torch.
    pause
    exit /b 1
)
set "KERAS_BACKEND=torch"
set "PYTHONIOENCODING=utf-8"
set "TRAIN_OUTPUT_DIR=%~dp0runs\%~1"
"%~dp0.venv-torch\Scripts\python.exe" "%~dp0train.py"
echo.
echo Entrenamiento terminado. Resultados en %TRAIN_OUTPUT_DIR%
pause
