python -m venv venv

venv\Scripts\activate

pip install indic-transliteration

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

 uvicorn main:app --reload


 uvicorn main:app --host 0.0.0.0 --port 8000


pip install -r requirement.txt


### NEW Way 
:: 1. Create virtual environment
python -m venv venv

:: 2. Activate it
venv\Scripts\activate

:: 3. Upgrade pip
python -m pip install --upgrade pip

:: 4. Install PyTorch FIRST (CPU version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: 5. Install everything else
pip install -r requirements.txt

:: 5. Freeze exact versions (do this once after successful install)
pip freeze > requirements-freeze.txt

:: 6. Run the app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

:: 6. Run the app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload