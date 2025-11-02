# User Identification from Behavioral Traces

This project focuses on identifying users from their behavioral traces in **Copilote**, an Enterprise Resource Planning (ERP) software developed by Infologic for the agro-food industry.

## Overview

Copilote is used by hundreds of users daily, generating large volumes of usage traces that reflect their work behaviors and habits. This project leverages data science techniques to automatically identify users based on their interaction patterns with the software.

The approach combines:
- **Data analysis** and exploration
- **Behavioral modeling** 
- **Supervised classification** using machine learning

## Project Structure

```
Project/
├── data/
│   ├── train.csv          # Training dataset (3279 sessions)
│   ├── test.csv           # Test dataset for predictions
│   └── submission.csv     # Generated predictions
│
├── pipeline/
│   ├── create_df.py              # Load CSV datasets
│   ├── build_actions.py          # Extract and encode user actions
│   ├── build_patterns.py          # Extract patterns from actions
│   ├── build_browsers.py          # Extract browser information
│   ├── build_encoded_data.py     # Tokenize sessions
│   └── build_actionspatterns.py  # Transform to feature matrix
│
├── visualise/
│   ├── visualise_actions.py           # Action filtering utilities
│   ├── visualise_patterns.py          # Pattern analysis
│   └── visualise_session_ratios.py    # Session temporal analysis
│
├── artifacts/
│   ├── actions.json      # Action-to-ID mapping
│   ├── patterns.txt      # Pattern-to-ID mapping
│   └── browsers.txt      # Browser-to-ID mapping
│
└── Data_Analysis_Report.ipynb  # Main analysis report (see below)
```

## Main Report

**`Data_Analysis_Report.ipynb`** contains the complete analysis report. This Jupyter notebook includes:

- **Data exploration**: Browser distribution, temporal patterns, action structure
- **Feature engineering**: Extraction of actions, patterns, and their combinations
- **Data preprocessing**: Tokenization and encoding of sessions into numerical features
- **Model training**: XGBoost classifier for multi-class user identification
- **Results**: Model evaluation and predictions

The notebook serves as both the methodology documentation and the execution pipeline for reproducing the analysis.

## Data Description

The dataset represents **user sessions** with the following characteristics:

- **Each row** = one unique session
- **Columns**:
  - `util`: User identifier
  - `navigateur`: Browser used (Firefox, Google Chrome, Microsoft Edge, Opera)
  - `col3 ... colN`: Sequence of actions observed during the session

- **Actions**: User interactions such as:
  - Screen creation/selection
  - Button clicks
  - Form field input
  - Filtering/sorting operations
  - Dialog displays
  - And 34 unique action types in total

- **Patterns**: Contextual information attached to actions (extracted from delimiters: `()`, `<>`, `$$`)

- **Time discretization**: Actions are recorded in 5-second intervals (marked as `t5`, `t10`, `t15`, etc.)

## Methodology

1. **Data Loading**: Read and parse CSV files containing session traces
2. **Action Extraction**: Identify and categorize 34 unique action types
3. **Pattern Extraction**: Extract 701 unique patterns that provide context to actions
4. **Tokenization**: Convert actions and patterns to numerical IDs
5. **Feature Engineering**: Transform sessions into feature vectors (frequency of action-pattern combinations)
6. **Model Training**: Train XGBoost classifier on the feature matrix
7. **Evaluation**: Achieved **77% accuracy** on user identification

## Key Findings

- Each user consistently uses the same browser across sessions
- Temporal patterns (action timing) are user-specific and discriminative
- Action-pattern combinations effectively capture user behavioral signatures
- Browser type is a stable user characteristic

## Requirements

The project uses:
- `pandas` for data manipulation
- `xgboost` for classification
- `sklearn` for model evaluation
- Standard Python libraries (`json`, `re`, `os`)

## Usage

To reproduce the analysis:

1. Ensure the data files (`train.csv`, `test.csv`) are in the `data/` directory
2. Open and run the cells in `Data_Analysis_Report.ipynb` sequentially
3. The notebook will:
   - Generate artifacts (action/pattern/browser mappings)
   - Process and encode the data
   - Train the XGBoost model
   - Generate predictions in `data/submission.csv`

## Future Improvements

Potential enhancements identified:
- Incorporate sequential dependencies using Transformer models
- Better modeling of temporal patterns in action sequences
- Improved feature engineering to capture session-level behaviors

---

*This project was developed as part of a data science course focusing on user identification from behavioral traces.*

