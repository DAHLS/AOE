# AOE - Automatic Organization Estimator

A machine learning tool for automatically classifying academic publications by organizational affiliation at the University of Copenhagen.

## Overview

**AOE (Automatic Organization Estimator)** is developed by **Asbjørn Dahl** at the **Copenhagen University Library**. It is used during the **'Slutsøgning'** (final search) process to help distribute the workload of importing articles by KU authors between the different faculty library departments.

The system classifies publications into faculty categories:
- **STEM**: Faculty of Health and Medical Sciences, Faculty of Science
- **Non-STEM**: Faculty of Humanities, Faculty of Law, Faculty of Social Sciences, Faculty of Theology

## Features

- Text feature extraction using TF-IDF vectorization
- Multiple ML classifiers: SVM, Random Forest, Logistic Regression, kNN, Decision Tree, Gradient Boosting
- Hyperparameter tuning via GridSearchCV
- Model comparison and evaluation metrics (accuracy, precision, recall, F1-score)
- Excel-based input/output for easy integration with existing workflows

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Data

```bash
python AOE_preprocessing.py path/to/Org_dump.xlsx
```

Processes raw CURIS export data, extracts faculty affiliations, and prepares text features.

### 2. Train & Compare Models

```bash
# Train all models with default settings (uses most recent processed file)
python AOE_BoW-Extended.py

# Specify input file explicitly
python AOE_BoW-Extended.py --input data/my_specific_file.xlsx

# Train specific algorithms
python AOE_BoW-Extended.py --algo svm random_forest knn

# Enable hyperparameter tuning
python AOE_BoW-Extended.py --hyperpara y

# Save trained models
python AOE_BoW-Extended.py --savemodel y

# See all options
python AOE_BoW-Extended.py --help
```

### 3. Make Predictions

```bash
# Interactive mode (prompts for model/vectorizer selection)
python AOE_predictor.py path/to/new_data.xlsx

# Fully automated (specify model and vectorizer paths)
python AOE_predictor.py path/to/new_data.xlsx \
    --model models/AOE_kNN_20251021_024848.pkl \
    --vectorizer models/AOE_tfidf-bow.pkl

# Custom output directory
python AOE_predictor.py path/to/new_data.xlsx -o my_output/
```

## Command-Line Options

### AOE_BoW-Extended.py

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Path to preprocessed data file (default: latest in `data/`) |
| `--algo` | Algorithms to test (default: all) |
| `--feat` | Maximum TF-IDF features |
| `--savemodel` | Save models to file (y/n) |
| `--hyperpara` | Enable hyperparameter tuning (y/n) |
| `--production` | Production mode (no random state) |

### AOE_predictor.py

| Option | Description |
|--------|-------------|
| `data_path` | Path to input data Excel file |
| `--model`, `-m` | Path to model file (default: interactive selection) |
| `--vectorizer`, `-v` | Path to vectorizer file (default: interactive selection) |
| `--output-dir`, `-o` | Output directory (default: `output/`) |

## Project Structure

```
AOE/
├── AOE_preprocessing.py    # Data preprocessing
├── AOE_BoW-Extended.py     # Model training & comparison
├── AOE_predictor.py        # Prediction on new data
├── data/                   # Input/output data
├── models/                 # Saved models & vectorizers
├── output/                 # Prediction results
└── misc/                   # Configuration & auxiliary files
```

## License

See [LICENSE](LICENSE) file.

## Contact

Developed by [Asbjørn Dahl](asda@kb.dk), Copenhagen University Library
