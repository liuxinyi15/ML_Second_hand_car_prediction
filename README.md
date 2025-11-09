# ML_Second_hand_car_prediction

---

## Dataset Description and Machine Learning Problem

### Overview

The competition aims to **predict the transaction price of used cars** based on their various features.  
The dataset is provided by a major online used car trading platform and contains **over 400,000 transaction records**.  
It includes **31 columns (features)**, of which **15 are anonymized variables** (`v_0` to `v_14`).

To ensure fairness, the data has been split as follows:
- **Training set:** 150,000 samples  
- **Test set A:** 50,000 samples  
- **Test set B:** 50,000 samples  

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

### Machine Learning Task

This is a **supervised regression problem**, where the goal is to predict the **used car price (`price`)** based on all other available features.

The evaluation metric for the competition is **Mean Absolute Error (MAE)**, defined as:

$$
MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

where  
- $y_i$ = true price of the $i^{th}$ car  
- $\hat{y}_i$ = predicted price of the $i^{th}$ car  
- $n$ = total number of cars in the test set

A **lower MAE** indicates better model performance.

---

## Structure of my GitHub
```python
project-root/
├── .gitattributes                  # Git text and end-of-line normalization
├── .gitignore                      # Ignore unnecessary files (e.g., cache, temp, checkpoints)
│
├── best_model.joblib               # Saved trained model (Joblib format)
│
├── data_for_lr.csv                 # Cleaned dataset for Linear Regression model
├── data_for_lr.zip                 # Compressed version of the linear regression dataset
├── data_for_tree.csv               # Cleaned dataset for tree-based models
│
├── used_car_train_20200313.csv     # Original training dataset
├── used_car_testA_20200313.csv     # Test dataset A (used for model evaluation)
├── used_car_testA_20200313.csv.zip
├── used_car_testB_20200421.zip     # Test dataset B (additional data)
│
├── Step_1_EDA.ipynb                # Step 1: Exploratory Data Analysis (EDA)
├── Step_2_Feature_Engineering.ipynb# Step 2: Feature Engineering and Preprocessing
├── Step_3_Modeling.ipynb           # Step 3: Model training, tuning, and evaluation
│
├── README.md                       # Project documentation
```


This repository contains the full workflow for a **used car price prediction** project, including:
- **Data exploration and visualization**
- **Feature engineering and preprocessing**
- **Model training and evaluation**
- **Model saving and reuse**

The `Step_3_Modeling.ipynb` notebook provides a detailed record of the entire model development and learning process, from baseline models to the optimized final version.
