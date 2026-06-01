#!/bin/bash

echo "==================================================="
echo "  Installing environments for Aydin and CSBDeep"
echo "==================================================="

echo ""
echo "[1/3] Creating CSBDeep environment..."
python3.9 -m venv venv_csbdeep
source venv_csbdeep/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r ./requirements/csbdeep_requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=venv_csbdeep --display-name="Python (CSBDeep)"
deactivate

echo ""
echo "[2/3] Creating Noise2Self environment..."
python3.9 -m venv venv_aydin
source venv_aydin/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r ./requirements/aydin_requirements.txt
pip install tensorflow-macos==2.10.0 tensorflow-metal==0.6.0
python -c "import aydin.util.log.log as lg; path = lg.__file__.replace('.pyc', '.py'); txt = open(path, 'r', encoding='utf-8').read(); open(path, 'w', encoding='utf-8').write(txt.replace('c_scafold', 'c_scaffold'))"
pip install ipykernel
python -m ipykernel install --user --name=venv_aydin --display-name="Python (Aydin)"
deactivate

echo ""
echo "[3/3] Creating main environment..."
python3.9 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r ./requirements/project_requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=venv_main --display-name="Python (Main)"
deactivate

echo ""
echo "==================================================="
echo "  Installation finished!"
echo "==================================================="
read -n 1 -s -r -p "Press any key to continue..."
echo ""