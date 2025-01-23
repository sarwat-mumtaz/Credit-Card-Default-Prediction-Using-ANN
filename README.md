# Credit-Card-Default-Prediction-Using-ANN


This project uses an artificial neural network (ANN) to predict whether a credit card holder will default on their payment next month. The dataset contains client demographics, financial attributes, and payment history.

## Project Overview

### Steps Implemented:

1. **Data Cleaning**: Removed irrelevant columns, checked for duplicates, and ensured data integrity.
2. **Exploratory Data Analysis (EDA)**: Analyzed class distributions and correlations; visualized credit limits by default status.
3. **Data Preparation**: Scaled numerical features and split the dataset into training and testing sets.
4. **Model Development**:
   - Built a neural network with two hidden layers and dropout regularization.
   - Used `adam` optimizer and binary cross-entropy loss.
5. **Model Training**: Trained the ANN with early stopping to prevent overfitting.
6. **Evaluation**: Evaluated the model using metrics like accuracy, confusion matrix, and ROC-AUC.
7. **Model Saving**: Saved the trained model for future use.

## Dataset Information

- **Source**: [credit\_card\_default.csv]
- **Rows**: 30,000
- **Columns**: 25

### Key Features:

- **LIMIT\_BAL**: Credit limit of the cardholder.
- **SEX, EDUCATION, MARRIAGE**: Demographic attributes.
- **AGE**: Age of the cardholder.
- **PAY\_0 to PAY\_6**: Payment history over the past six months.
- **BILL\_AMT1 to BILL\_AMT6**: Monthly bill statements.
- **PAY\_AMT1 to PAY\_AMT6**: Monthly payments made.
- **default payment\_next\_month**: Target variable (1 = default, 0 = no default).

## Libraries Used

- **Data Handling**: Pandas, Numpy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: TensorFlow/Keras, scikit-learn

## Key Files

1. **Project Code**: Contains the full implementation of data preprocessing, EDA, model training, and evaluation.
2. **Trained Model**: Saved in the file `credit_card_default_ann_model.h5`.

## Instructions to Run

1. **Setup Environment**:

   - Install required libraries: `pip install pandas numpy matplotlib seaborn tensorflow scikit-learn`.

2. **Load Dataset**:

   - Place `credit_card_default.csv` in the working directory.

3. **Run the Script**:

   - Execute the Python script to train and evaluate the ANN model.

4. **Model Predictions**:

   - Use the saved model to make predictions on new data.

## Model Performance

- **Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC for assessing model discrimination ability.
- **Evaluation Results**:
  - Achieved competitive accuracy and ROC-AUC values.

## Future Improvements

- Hyperparameter tuning for better performance.
- Exploration of alternative machine learning models.
- Integration of additional features for improved prediction accuracy.



