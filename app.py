from flask import Flask, render_template, request
import pickle
from xgboost import XGBClassifier
import pandas 
from custom_encoders import CategoricalEncoder

app = Flask(__name__)


with open('bankCard_predict_modelF.pkl', 'rb') as f:
    model = pickle.load(f)

with open('bankCard_le_encod_targetF.pkl', 'rb') as g:
    label_encoder = pickle.load(g)

with open('bankCard_preprocess_pipeF.pkl', 'rb') as h:
    preprocess_pipeline = pickle.load(h)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    # input_features = [
    #     form_data['Attrition_Flag'], form_data['Gender'], form_data['Marital_Status'],
    #     form_data['Education_Level'], form_data['Income_Category'], form_data['Customer_Age'],
    #     form_data['Months_on_book'], form_data['Months_Inactive_12_mon'], form_data['Credit_Limit'],
    #     form_data['Total_Revolving_Bal'], form_data['Avg_Open_To_Buy'], form_data['Total_Amt_Chng_Q4_Q1'],
    #     form_data['Total_Trans_Amt'], form_data['Total_Trans_Ct'], form_data['Total_Ct_Chng_Q4_Q1'],
    #     form_data['Avg_Utilization_Ratio'],form_data['Total_Relationship_Count'],form_data['Dependent_count'],form_data['Contacts_Count_12_mon']
    # ]
    input_features = [
    form_data['Attrition_Flag'],
    int(form_data['Customer_Age']),
    form_data['Gender'],
    int(form_data['Dependent_count']),
     form_data['Education_Level'],
     form_data['Marital_Status'],
     form_data['Income_Category'],
    int(form_data['Months_on_book']),    
    int(form_data['Total_Relationship_Count']),
    int(form_data['Months_Inactive_12_mon']),
    int(form_data['Contacts_Count_12_mon']),
    float(form_data['Credit_Limit']), 
    float(form_data['Total_Revolving_Bal']), 
    float(form_data['Avg_Open_To_Buy']),
    float(form_data['Total_Amt_Chng_Q4_Q1']),  # Convert to float
    float(form_data['Total_Trans_Amt']),  # Convert to float
    int(form_data['Total_Trans_Ct']),  # Convert to int
    float(form_data['Total_Ct_Chng_Q4_Q1']),  # Convert to int
    float(form_data['Avg_Utilization_Ratio'])
]
    


    current_card = form_data['Current_Card']  
    input_df = pandas.DataFrame([input_features], columns=[
        'Attrition_Flag','Customer_Age', 'Gender','Dependent_count','Education_Level', 'Marital_Status', 'Income_Category',
        'Months_on_book','Total_Relationship_Count', 'Months_Inactive_12_mon','Contacts_Count_12_mon', 'Credit_Limit',
        'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
    ])

    print(input_df.dtypes)
    
    input_df[['Customer_Age',"Dependent_count", "Months_on_book","Total_Relationship_Count","Months_Inactive_12_mon",
    "Contacts_Count_12_mon","Total_Trans_Ct"]] = input_df[['Customer_Age',"Dependent_count", "Months_on_book","Total_Relationship_Count","Months_Inactive_12_mon",
    "Contacts_Count_12_mon","Total_Trans_Ct"
    ]].applymap(int)

    input_df[['Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']]= input_df[['Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']].applymap(float)
    
    input_df = preprocess_pipeline.transform(input_df)
    prediction = model.predict(input_df)
    predicted_card = label_encoder.inverse_transform(prediction)[0]

    if current_card != predicted_card:
        if (current_card == 'Blue' and predicted_card in ['Silver', 'Gold', 'Platinum']) or (current_card == 'Silver' and predicted_card in ['Gold', 'Platinum']) or (current_card == 'Gold' and predicted_card in ['Platinum']):
            message = f"Upgrade to {predicted_card}"
        else:
            message = "No upgrade required"
    else:
        message = "No upgrade required"

    return render_template('result.html', prediction_message=message)
    # return "ok"
if __name__ == '__main__':
    app.run(debug=True)
