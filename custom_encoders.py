# In custom_encoders.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.categorical_columns = ['Attrition_Flag', 'Gender', 'Marital_Status', 
                                     'Education_Level', 'Income_Category']
    
    def fit(self, X, y=None):
        for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.categorical_columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy
