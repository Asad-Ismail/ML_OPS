# Machine Learning Pipeline Manager

## Overview
AWS ML OPS pipeline can be found in aws_MLOPS directory along with the code

This project implements a Machine Learning Pipeline Manager using Python. It contain MVP to describe how model_management can work

## Features
- **Data Loading and Preparation**: Automated downloading and preparation of datasets.
- **Pipeline Management**: Easily configurable machine learning pipelines.
- **Model Training and Evaluation**: Training and evaluation of models with precision, recall, and F1 score metrics.
- **Inference**: Perform inference on new data using trained models.
- **Artifacts**:  Model evaluation  metrics are also logged  which can be retrieved later
- **Data Monitoring**: Simple data drift detection by comparing new data statistics against training data statistics.

## Requirements

Install requriements using 

```bash
pip install -r requirements.txt
```

## Test 

To test model_management class 


```bash
python model_management.py
```

It will also download sample credit fraud dataset train model and 