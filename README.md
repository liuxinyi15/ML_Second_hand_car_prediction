# ML_Second_hand_car_prediction

---

## Overview and Dataset Description

### Overview

This repository implements a complete supervised Machine Learning pipeline to predict used car prices based on tabular data from the Used Car Dataset.
The goal is to build a regression model capable of predicting price using various vehicle attributes.
The dataset is provided by a major online used car trading platform and contains **over 400,000 transaction records**.  
It includes **31 columns (features)**, of which **15 are anonymized variables** (`v_0` to `v_14`).

Additionally, identifiers such as `name`, `model`, `brand`, and `regionCode` have been **desensitized (anonymized)** for privacy protection.

---

### Objective

Develop and optimize a regression model capable of accurately predicting the transaction prices of used cars using both the **explicit features** (e.g., power, kilometer, body type) and the **anonymous features** $(v_0 - v_{14})$, while minimizing the MAE score on the test sets.

---

### Feature Description

| Feature Name | Description |
|---------------|-------------|
| SaleID | Unique ID of the sales record |
| name | Vehicle code |
| regDate | Vehicle registration date |
| model | Model code |
| brand | Brand of the vehicle |
| bodyType | Body type (e.g., sedan, SUV, etc.) |
| fuelType | Type of fuel used (e.g., gasoline, diesel, etc.) |
| gearbox | Type of transmission (manual/automatic) |
| power | Engine power |
| kilometer | Total mileage (in kilometers) |
| notRepairedDamage | Indicates whether there is unrepaired damage |
| regionCode | Encoded region where the vehicle is sold |
| seller | Seller type |
| offerType | Offer type |
| creatDate | Date when the ad was published |
| price | **Target variable** – transaction price of the vehicle |
| $v_0$ to $v_{14}$ | Anonymous numerical features representing hidden patterns or composite variables |

---

## Structure of my GitHub
```bash
│
├── data/
│   └── used_car_train_20200313.csv        # Raw training dataset
│
├── feature/
│   ├── feature_engineering.py             # Feature engineering functions
│   ├── process_data.py                    # Processing (fix_missing, numericalize, process_df)
│   └── transform_to_log.py                # Log transform of the target variable
│
├── model/
│   ├── main.ipynb                         # Main notebook (training + validation + inference)
│   └── opt.db                             # Optuna database storing tuning results
│
├── user_data/
│   ├── best_model.joblib                  # Trained Random Forest model
│   ├── XGBoost_Model.joblib               # Trained XGBoost model
│   ├── processed_data.csv                 # For visualization
│   ├── processed_feature.csv              # For visualization
│   └── price_log_transform.csv            # Cleaned + transformed training data
│
├── cisualization/
│   ├── app.py                             # Streamlit visualization
├── README.md
└── requirements.txt
```


This repository contains the full workflow for a **used car price prediction** project, including:
- Data preprocessing
- Feature engineering
- Outlier handling
- Log transformation of target
- Train/validation split
- Model training (Random Forest+XGBoost)
- Hyperparameter optimization using Optuna

---
## Data preprocessing pipeling-Inside `feature/` folder

- `fix_missing`
  - Fills numerical missing values using median
  - Adds an additional indicator column `xxx_na` when appropriate (`pd.get_dummies()` -> `process_df`)
- `numericalize`
    - Converts categorical variables into integer codes
- `process_df`
    - Remove ignored fields
    - Applying custom preprocessing functions
    - Handling missing values
    - Converting categorical variables
    - Genrating dummy vairables  (`pd.get_dummies()`)
- `outlier removal`
    - Use method based on quantile, remove values outside the quantile range 0.1-0.9
- `log transformation`
    `price -> price_log` to reduce skewness and stabilize variance

---
## Feature engineering
- used_time
- date-based features
- interaction or aggregated variables (v_0, v_1, …, v_14)

---
## How to run the project

- Step 1 - Install dependencies
  ``` bash
  pip install -r requirements.txt
  ```
- Step 2 - Open Jupyter Notebook in `main/` folder
- Step 3 - Run the full pipeline `model/main.ipynb`
- Step 4 - Explore the visualization
  ``` bash
  streamlit run visualization/app.py
  ``` 
