# Physics-Informed Soil Moisture Modeling

End-to-end workflow for soil moisture prediction using a physics-informed Bi-LSTM model.

This project is organized into three notebook phases:

- `Phase1_Extraction.ipynb`: data extraction and initial alignment
- `Phase2_Preprocessing.ipynb`: feature engineering and dataset preparation
- `Phase3_PhysicsModel.ipynb`: model training, evaluation, and artifact export

Core model/loss implementations live in:

- `physics_model.py`

## Project Structure

- `assets/roi_aligned_timeseries_2021_2025.csv`: aligned input dataset (local, ignored)
- `physics_sm_model.pt`: trained model checkpoint (generated artifact)
- `assets/`: generated plots used for documentation
- `requirements.txt`: Python dependencies

## Results Snapshot

### Training Dynamics

![Training Curves](assets/training_curves.png)

### Monsoon Holdout Evaluation

![Monsoon Holdout Test Results](assets/test_results.png)

### Full Timeline Prediction (2021-2025)

![Full Year Results](assets/full_year_results.png)

## Quick Start

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

1. `Phase1_Extraction.ipynb`
2. `Phase2_Preprocessing.ipynb`
3. `Phase3_PhysicsModel.ipynb`

## Reproducibility Notes

- The model in Phase 3 uses train/validation split with monsoon holdout for testing.
- Generated artifacts are ignored by default via `.gitignore`.
- Keep raw input data versioned if it is small and shareable; avoid committing large binaries.

## Suggested GitHub Topics

`soil-moisture` `lstm` `physics-informed-ml` `pytorch` `hydrology` `remote-sensing`

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
