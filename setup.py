# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install base requirements
pip install torch torchvision numpy pandas matplotlib jupyter
pip install -U scikit-learn tqdm pillow