@echo off
echo === Setting up Python environment ===

:: Upgrade pip
python -m pip install --upgrade pip

echo.
echo === Installing CUDA‑enabled PyTorch and Vision ===
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo.
echo === Installing remaining dependencies ===
pip install -r requirements.txt

echo.
echo Installation complete!
echo Run your app with: streamlit run app.py
pause
