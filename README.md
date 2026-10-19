# IoT S1: Human Activity Recognition

Classifies walking, standing, sitting, stand-to-sit and sit-to-stand from
smartphone accelerometer and gyroscope data (UCI HAPT dataset, then our own recordings).

## Setup
1. Clone this repo with GitHub Desktop.
2. In a terminal in the project folder:
```
   python -m venv venv
   venv\Scripts\Activate.ps1
   pip install -r requirements.txt
```
3. In VS Code, select the `venv` interpreter (Ctrl+Shift+P → Python: Select Interpreter).
4. Download the HAPT dataset and put the contents of `RawData` in `data/hapt_raw/RawData/`.

## Data
The `data/` folder is not tracked by Git. Never commit recordings: they are
personal motion data and must be stored securely.

## Structure
- `src/preprocess.py`: filtering, windowing and features, shared by all notebooks
- `notebooks/`: exploration, dataset building and model training
