#!/bin/bash

echo "==================================================="
echo "  Installing environments (Aydin + CSBDeep)"
echo "==================================================="

echo ""
echo "[1/3] Creating CSBDeep environment..."
python3.9 -m venv venv_csbdeep
source venv_csbdeep/bin/activate
python -m pip install --upgrade pip
pip install -r ./requirements/csbdeep_requirements.txt
deactivate

echo ""
echo "[2/3] Creating Noise2Self environment..."
python3.9 -m venv venv_aydin
source venv_aydin/bin/activate
python -m pip install --upgrade pip
pip install -r ./requirements/aydin_requirements.txt
deactivate

echo ""
echo "[3/3] Creating main environment..."
python3.9 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r ./requirements/project_requirements.txt
deactivate

echo ""
echo "==================================================="
echo "  Installation finished!"
echo "==================================================="
read -n 1 -s -r -p "Press any key to continue..."
echo ""