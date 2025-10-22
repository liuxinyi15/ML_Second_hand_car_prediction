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

Develop and optimize a regression model capable of accurately predicting the transaction prices of used cars using both the **explicit features** (e.g., power, kilometer, body type) and the **anonymous features** ($v_0$–$v_{14}$), while minimizing the MAE score on the test sets.

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
