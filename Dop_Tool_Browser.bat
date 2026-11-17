@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "CONDA_SSL_VERIFY=truststore"
set "PYTHONPATH=%~dp0runtime_compat"
if not exist .venv312\python.exe if not exist .venv312\Scripts\python.exe (
  if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    call "%USERPROFILE%\anaconda3\condabin\conda.bat" create -y -p .venv312 python=3.12 pip
  ) else (
    py -3.12 -m venv .venv312
  )
  if errorlevel 1 (echo Khong tao duoc moi truong Python 3.12. & pause & exit /b 1)
  if exist .venv312\Scripts\python.exe (
    "%USERPROFILE%\anaconda3\python.exe" -m pip --python .venv312 install pip
    if errorlevel 1 (echo Khong cai duoc pip vao moi truong Python. & pause & exit /b 1)
  ) else if exist .venv312\python.exe (
    .venv312\python.exe -m ensurepip --upgrade
    if errorlevel 1 (echo Khong cai duoc pip vao moi truong Python. & pause & exit /b 1)
  )
)
if exist .venv312\python.exe (set "DOP_PYTHON=.venv312\python.exe") else (set "DOP_PYTHON=.venv312\Scripts\python.exe")
if exist "%USERPROFILE%\anaconda3\Library\bin" set "PATH=%USERPROFILE%\anaconda3\Library\bin;%PATH%"
%DOP_PYTHON% -m pip install -q --upgrade pip
%DOP_PYTHON% -m pip install -q -r requirements.txt
if errorlevel 1 (echo Cai thu vien that bai. Hay kiem tra ket noi mang va Python 3.12. & pause & exit /b 1)
%DOP_PYTHON% -m streamlit run web_app.py --browser.gatherUsageStats false
pause
