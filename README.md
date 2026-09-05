
# 🌱 Crop Advisor AI

### An Explainable AI-Powered Crop Recommendation System

Crop Advisor AI is a machine learning application designed to help farmers make informed crop-selection decisions based on **soil nutrients and environmental conditions**.

The system uses an **XGBoost multiclass classification model** to predict suitable crops from agricultural conditions including nitrogen, phosphorus, potassium, temperature, humidity, soil pH, rainfall, and soil texture.

The model is integrated into a **Streamlit web application** that provides the top three predicted crops and human-readable explanations based on historical crop-condition patterns in the dataset.

---

## 🚜 Problem Statement

Farmers often make crop-selection decisions based on experience, traditional knowledge, or generalized agricultural recommendations. However, crop performance can vary depending on soil and environmental conditions.

Crop Advisor AI aims to provide a data-driven decision-support tool that helps farmers identify crops that are statistically compatible with their available growing conditions.

---

## 💡 Solution

Crop Advisor AI combines machine learning with an explainable recommendation layer.

Users provide:

* 🌱 Nitrogen (N)
* 🌱 Phosphorus (P)
* 🌱 Potassium (K)
* 🌡️ Temperature
* 💧 Humidity
* 🧪 Soil pH
* 🌧️ Rainfall
* 🪨 Soil Texture

The trained XGBoost model processes these conditions and produces predictions for multiple crop classes.

The application then:

1. Identifies the top three predicted crops.
2. Displays the model's predicted probability for each crop.
3. Compares the user's conditions with historical ranges observed for the recommended crops.
4. Provides human-readable reasons explaining why each crop was predicted.

---

## 🤖 Machine Learning Approach

This project uses **XGBoost (Extreme Gradient Boosting)** for multiclass crop classification.

### Features

The model uses eight agricultural features:

| Feature        | Description           |
| -------------- | --------------------- |
| `N`            | Nitrogen level        |
| `P`            | Phosphorus level      |
| `K`            | Potassium level       |
| `temperature`  | Temperature in °C     |
| `humidity`     | Relative humidity     |
| `ph`           | Soil pH               |
| `rainfall`     | Rainfall              |
| `soil_texture` | Soil texture category |

### Target

The target variable is:

```text
crop_label
```

which represents the crop to be recommended.

---

## 🔧 Data Preprocessing

### Target Encoding

The crop labels are categorical text values. `LabelEncoder` is used to convert the target crop labels into numerical class IDs for model training.

The same encoder is saved and used during inference to convert predicted class IDs back into human-readable crop names.

### Categorical Feature Encoding

`soil_texture` is a categorical input feature, so **One-Hot Encoding** is applied.

This prevents the model from interpreting soil categories as having an artificial numerical order.

The preprocessing and model are combined in a Scikit-learn `Pipeline` to ensure that the same preprocessing is applied during both training and prediction.

---

## 🧠 Model Architecture

The machine learning pipeline follows this process:

```text
Agricultural Dataset
        ↓
Data Cleaning
        ↓
Feature / Target Separation
        ↓
Target Label Encoding
        ↓
Train-Test Split
        ↓
One-Hot Encoding of Soil Texture
        ↓
XGBoost Classifier
        ↓
Stratified K-Fold Cross-Validation
        ↓
Final Model
        ↓
Streamlit Application
```

---

## 📊 Model Validation

The dataset was divided into training and testing sets using an **80/20 stratified split**.

To evaluate the consistency of the model during development, **5-fold Stratified Cross-Validation** was applied to the training data.

### Cross-Validation Results

| Fold                   |                   Accuracy |
| ---------------------- | -------------------------: |
| Fold 1                 |                    100.00% |
| Fold 2                 |                    100.00% |
| Fold 3                 |                    100.00% |
| Fold 4                 |                    99.375% |
| Fold 5                 |                    100.00% |
| **Mean**               |                **99.875%** |
| **Standard Deviation** | **0.25 percentage points** |

The results demonstrate highly consistent performance across the five stratified validation folds.

> **Note:** Cross-validation accuracy is different from the final holdout test accuracy. The 20% test set is kept separate for final evaluation.

---

## 🔍 Explainability

The application includes a human-readable explanation layer.

After the XGBoost model generates its predictions, the application compares the user's farm conditions with historical ranges observed for each predicted crop.

For example, the system can identify whether:

* rainfall falls within the observed range for a crop;
* humidity is within the observed range;
* soil texture matches historical samples;
* nitrogen, phosphorus, and potassium levels fall within observed ranges.

This allows the application to provide more context than simply displaying a crop name.

### Important distinction

The percentages displayed by the model represent **predicted probabilities from XGBoost's `predict_proba()`**, not a direct agronomic suitability percentage.

For example:

```text
Tomato — 12.5% model probability
Soybean — 12.0% model probability
Pepper — 11.9% model probability
```

The explanation layer is separate from these model probabilities.

---

## 🖥️ Application

The web application was developed using **Streamlit**.

Users can enter their farm conditions through an interactive interface and receive:

### 🏆 Top Crop Predictions

The application displays the three crops with the highest model-predicted probabilities.

### 📋 Farm Condition Summary

The user's submitted agricultural conditions are displayed for easy reference.

### 💡 Crop Explanation

The application provides human-readable explanations based on historical crop-condition patterns in the dataset.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation
* **Scikit-learn** — preprocessing, model pipeline, train-test splitting and cross-validation
* **XGBoost** — machine learning classification
* **Joblib** — model and encoder serialization
* **Streamlit** — web application
* **NumPy** — numerical operations

---

## 📁 Project Structure

```text
crop-advisory-system/
│
├── app.py
├── crop_prediction_model2.pkl
├── crop_label_encoder.pkl
├── original_dataset.csv
├── requirements.txt
└── README.md
```

### File Descriptions

| File                         | Purpose                                         |
| ---------------------------- | ----------------------------------------------- |
| `app.py`                     | Streamlit application                           |
| `crop_prediction_model2.pkl` | Trained XGBoost pipeline                        |
| `crop_label_encoder.pkl`     | Label encoder used for crop classes             |
| `original_dataset.csv`       | Dataset used for crop profiles and explanations |
| `requirements.txt`           | Python dependencies                             |
| `README.md`                  | Project documentation                           |

---

## 🚀 Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/crop-advisory-system.git
cd crop-advisory-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🌐 Live Application

**Live Demo:**
https://crop-advisory-system-kadai.streamlit.app/

---

## ⚠️ Limitations

The system is intended as a **decision-support tool**, not a replacement for professional agricultural advice.

The quality of its recommendations depends on:

* the quality and representativeness of the training dataset;
* the range of crops represented in the dataset;
* the accuracy of the farmer's input values;
* environmental conditions that may vary by location and season.

The current explanation system uses historical data ranges to provide context and does not represent causal relationships between individual features and crop outcomes.

---

## 🔮 Future Improvements

Future versions could include:

* 📍 Location-specific recommendations using geospatial and climate data
* 🌦️ Real-time weather and rainfall information
* 🧪 Integration with soil-testing services
* 🤖 SHAP-based feature attribution for deeper model explainability
* 📱 Mobile-friendly deployment
* 🌾 Additional crop varieties and regional datasets
* 🗣️ Local-language and voice-based recommendations
* 📈 Historical yield prediction
* 👨‍🌾 Personalized farming recommendations

---

## 🎯 Project Goal

Crop Advisor AI aims to make agricultural decision-making more **data-driven, accessible, and explainable**, helping farmers select crops based on the conditions available on their farms.

> **From farm conditions to data-driven crop recommendations. 🌱**
