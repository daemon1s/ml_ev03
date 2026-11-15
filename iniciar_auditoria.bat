@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0auditoria.env.bat" call "%~dp0auditoria.env.bat"
for %%V in (DATASET_DIR AUDIT_WORK_DIR AUDIT_OLLAMA_URL AUDIT_MODEL) do (
    if not defined %%V (
        echo Falta configurar %%V. Copia auditoria.env.example.bat como auditoria.env.bat y completa los valores.
        pause
        exit /b 1
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo Falta el entorno virtual en .venv. Sigue las instrucciones del README.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m detector.audit_gui
if errorlevel 1 pause
