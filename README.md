Overview
-------------------------------------------------------------
This project delivers an XGBoost based Machine-Learning model which is aimed at recommending if a customer with given economic fundamentals is statistically likely to upgrade his credit card type.

Business Use-Case
-------------------------------------------------------------
Instead of using a random exhaustive approach of calling customers, banks can use this model to:
- Make sure that only those customers are approached for credit-card upgradation who are statistically likely to upgrade.
- This not only improves the banks efficiency in targeting the right customers saving them time and resources but also enhances customer satisfaction by reducing the number of irrelevant calls from banks for credit-card upgradations that they are not interested in.

Dataset Features:
-------------------------------------------------------------

- Customer demographics (age, gender, education, marital status)
- Account information (months on book, credit limit)
- Transaction patterns (total amount, frequency)
- Financial behavior (revolving balance, utilization ratio)
- Relationship metrics (inactive months, contact frequency)

Models Implemented:
-------------------------------------------------------------
1. Logistic Regression (Baseline Model)
- Accuracy: 93.39%
- F1 Score: 91.95%

2. Random Forest Classifier
- Accuracy: 94.03%
- F1 Score: 92.34%
  
3. Support Vector Machine (with different kernels)
- Linear Kernel: 92.74% accuracy
- RBF Kernel: 93.34% accuracy
- Polynomial Kernel: 93.34% accuracy

4. Gradient Boosting Classifier
- Accuracy: 96.40%
- F1 Score: 96.52%

5. XGBoost Classifier (Best Performing)
- Accuracy: 95.90%
- F1 Score: 96.51%
- Precision: 97.30%

Installation
-------------------------------------------------------------
1. Clone the Repository using:
    ``` 
        git clone https://github.com/AbdulHaq1503/CustomerCreditCardTypePrediction_ML
        cd credit-card-upgrade-prediction
    ```
2. Install python version mentioned in the python_version document
3. Install the required dependencies:  
    Note: Ensure you have pip and a Python environment (e.g., virtualenv or Conda) set up before running this command.
    ```
        pip install -r requirements.txt
    ```
4. Model Deployment
The final model and preprocessing pipeline are saved as:
- Model: bankCard_predict_model.pkl
- Label Encoder: bankCard_le_encod_target.pkl
- Preprocessing Pipeline: bankCard_preprocess_pipe.pkl

