@echo off
echo ===================================================
echo   Installing environments (Aydin + CSBDeep)
echo ===================================================

echo.
echo [1/3] Creating CSBDeep environment...
py -3.9 -m venv venv_csbdeep
call venv_csbdeep\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r .\requirements\csbdeep_requirements.txt
call venv_csbdeep\Scripts\deactivate.bat

echo.
echo [2/3] Creating Noise2Self environment...
py -3.9 -m venv venv_aydin
call venv_aydin\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r .\requirements\aydin_requirements.txt
pip install tensorflow==2.10.0
call venv_aydin\Scripts\deactivate.bat

echo.
echo [3/3] Creating main environment...
py -3.9 -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r .\requirements\project_requirements.txt
call venv\Scripts\deactivate.bat

echo.
echo ===================================================
echo   Installation finished!
echo ===================================================
pause