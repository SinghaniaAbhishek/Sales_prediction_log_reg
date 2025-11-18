# 📊 Sales Prediction Using Logistic Regression# Sales Prediction using Logistic Regression



> A machine learning project to predict customer purchase behavior using binary classification## Introduction

This project's main goal is to predict sales outcomes using logistic regression. By analyzing the provided dataset, we build a predictive model to classify sales performance based on various features.

![Status](https://img.shields.io/badge/Status-Active-success)

![Python](https://img.shields.io/badge/Python-3.7+-blue)## Dataset

![License](https://img.shields.io/badge/License-MIT-green)The dataset used in this project is `DigitalAd_dataset.csv`. It contains the following columns:

- **Target**: The target variable indicating sales outcomes (e.g., 0 for no sale, 1 for sale).

---

## Methodology

## 📌 Overview1. **Data Preprocessing**: Cleaning and preparing the dataset for analysis.

2. **Exploratory Data Analysis (EDA)**: Understanding the data through visualization and statistical measures.

This project leverages **Logistic Regression**, a powerful classification algorithm, to predict whether a customer will make a purchase based on demographic features. The model analyzes customer age and salary data to classify purchasing behavior with high accuracy.3. **Model Training**: Using logistic regression to train the model on the dataset.

4. **Evaluation**: Assessing the model's performance using metrics such as accuracy, precision, recall, and F1-score.

**Key Objective:** Build and evaluate a predictive model that can accurately forecast customer purchase decisions.

## How to Use

---1. Clone the repository:

   ```bash

## 🎯 Features   git clone https://github.com/Keshab1257/Sales_prediction_log_reg.git

   ```

✅ Data preprocessing and feature scaling  2. Navigate to the project directory:

✅ Exploratory Data Analysis (EDA) with visualizations     ```bash

✅ Binary classification using Logistic Regression     cd Sales_prediction_log_reg

✅ Model evaluation with accuracy, precision, recall & F1-score     ```

✅ Interactive customer purchase prediction  3. Run the Jupyter Notebook `sales_pred.ipynb` to see the step-by-step implementation.

✅ Confusion matrix analysis  4. Alternatively, execute the Python script `sales_pred.py` for a streamlined version of the analysis.

✅ Both Jupyter Notebook and Python script implementations  

## Acknowledgments

---- The dataset was sourced from [source name].

- Inspiration for this project came from [source of inspiration].
## 📂 Project Structure

```
Sales_prediction_log_reg/
├── README.md                          # Project documentation
├── sales_pred.ipynb                   # Jupyter Notebook with detailed analysis
├── sales_pred.py                      # Standalone Python script
└── DigitalAd_dataset.csv             # Dataset (402 records)
```

---

## 📊 Dataset

**File:** `DigitalAd_dataset.csv`

| Column | Type | Description |
|--------|------|-------------|
| **Age** | Integer | Customer age in years |
| **Salary** | Integer | Customer annual salary in currency units |
| **Status** | Binary | Purchase outcome (0 = No Purchase, 1 = Purchase) |

**Dataset Statistics:**
- Total Records: 402
- Features: 2 (Age, Salary)
- Target Classes: 2 (Binary Classification)
- Class Distribution: Balanced dataset

---

## 🔧 Technology Stack

- **Python 3.7+**
- **Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` & `seaborn` - Data visualization
  - `jupyter` - Interactive notebook environment

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.7 or higher installed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<owner name>/Sales_prediction_log_reg.git
   cd Sales_prediction_log_reg
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook sales_pred.ipynb
   ```

4. **Or run the Python script:**
   ```bash
   python sales_pred.py
   ```

---

## 📈 Methodology

### Step 1: Data Preprocessing
- Load the dataset using pandas
- Handle missing values (if any)
- Explore dataset shape and structure

### Step 2: Exploratory Data Analysis (EDA)
- Generate pair plots to visualize relationships
- Analyze feature distributions
- Identify patterns in customer behavior

### Step 3: Data Preparation
- Split data into training (75%) and testing (25%) sets
- Apply StandardScaler for feature normalization
- Ensure model convergence and optimal performance

### Step 4: Model Training
- Train Logistic Regression classifier
- Fit the model on training data
- Learn optimal decision boundaries

### Step 5: Model Evaluation
- Generate predictions on test data
- Calculate Confusion Matrix
- Compute Accuracy, Precision, Recall, and F1-Score

### Step 6: Prediction
- Make predictions for new customers
- Determine purchase probability

---

## 📊 Model Performance

The trained model provides:
- **Confusion Matrix:** Shows True Positives, True Negatives, False Positives, and False Negatives
- **Accuracy:** Overall model correctness percentage
- **Classification Metrics:** Precision, Recall, and F1-Score for detailed performance analysis

---

## 💡 Usage Example

### Interactive Prediction
```python
# Enter customer details
age = 35
salary = 85000

# Model will predict purchase behavior
# Output: "Customer will Buy" or "Customer won't Buy"
```

---

## 📝 Project Workflow

```
Data Loading
    ↓
EDA & Visualization
    ↓
Data Splitting (Train/Test)
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Customer Prediction
```

---

## 🎓 Key Learnings

- **Binary Classification:** Understanding supervised learning for two-class problems
- **Feature Scaling:** Importance of normalization in logistic regression
- **Model Evaluation:** Using confusion matrix and accuracy metrics
- **Prediction:** Making real-world predictions with trained models

---

## 📌 Important Notes

- This project assumes a cleaned, well-structured dataset
- Logistic Regression works best with linearly separable data
- Feature scaling (StandardScaler) is crucial for optimal model performance
- The model is trained with random_state=0 for reproducibility

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📧 Support & Feedback

For questions, suggestions, or feedback, please feel free to open an issue on the GitHub repository.

---

**⭐ If you found this project helpful, please consider giving it a star!**
