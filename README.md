# Machine Learning Pipeline Manager

## Overview
This project implements a Machine Learning Pipeline Manager using Python. It contain MVP to describe how model_management can work

## Features
- **Data Loading and Preparation**: Automated downloading and preparation of datasets.
- **Pipeline Management**: Easily configurable machine learning pipelines.
- **Model Training and Evaluation**: Training and evaluation of models with precision, recall, and F1 score metrics.
- **Inference**: Perform inference on new data using trained models.
- **Artifacts**:  Model evaluation  metrics are also logged  which can be retrieved later
- **Data Monitoring**: Simple data drift detection by comparing new data statistics against training data statistics.

## Requirements
The project is built using Python and relies on several libraries including Pandas, Scikit-Learn, XGBoost, Pydantic, and tqdm. To install the required libraries, run:


```bash
pip install -r requirements.txt
```