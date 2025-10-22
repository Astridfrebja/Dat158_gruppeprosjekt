from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import os

# --- Oppsett ---
app = Flask(__name__)

# --- 1. Last inn Modell ---
MODEL_PATH = 'titanic_rf_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Titanic Random Forest-modell lastet inn OK.")
except Exception as e:
    print(f"Feil ved lasting av modell: {e}")
    model = None
    
# --- Rute for Hjemmeside og Prediksjon ---
@app.route('/', methods=['GET', 'POST'])
def predict_survival():
    prediction_result = None
    
    # Standardverdier for input-feltene (brukes ved første lasting)
    user_inputs = {
        'pclass': '3',  # Lagres som streng for å matche HTML-form data
        'age': '30.0', 
        'fare': '50.00', 
        'sibsp': '0', 
        'parch': '0', 
        'sex': 'female', 
        'embarked': 'S'
    }

    if request.method == 'POST' and model is not None:
        
        # 1a. LAGRING AV RÅDATA: Henter alle inputs og lagrer dem i user_inputs
        # Dette gjør at vi kan sende dem tilbake til HTML for å unngå nullstilling
        user_inputs['pclass'] = request.form.get('pclass', user_inputs['pclass'])
        user_inputs['age'] = request.form.get('age', user_inputs['age'])
        user_inputs['fare'] = request.form.get('fare', user_inputs['fare'])
        user_inputs['sibsp'] = request.form.get('sibsp', user_inputs['sibsp'])
        user_inputs['parch'] = request.form.get('parch', user_inputs['parch'])
        user_inputs['sex'] = request.form.get('sex', user_inputs['sex'])
        user_inputs['embarked'] = request.form.get('embarked', user_inputs['embarked'])

        try:
            # 1b. VALIDERTE INPUTS: Konverterer strenger (fra user_inputs) til tall for modellen
            
            inputs_for_model = {}
            inputs_for_model['pclass'] = int(user_inputs['pclass'])
            inputs_for_model['age'] = float(user_inputs['age'])
            inputs_for_model['fare'] = float(user_inputs['fare'])
            inputs_for_model['sibsp'] = int(user_inputs['sibsp'])
            inputs_for_model['parch'] = int(user_inputs['parch'])
            
            sex = user_inputs['sex']
            embarked = user_inputs['embarked']
            
            # --- 2. Dataforberedelse (MÅ MATCHE TRENTE FEATURES!) ---
            
            # **DEN FIKSEDE FEATURE_ORDER** (Basert på Colab utskriften: Age, SibSp, Parch, Fare, Sex_male, Embarked, Pclass)
            FEATURE_ORDER = ['Age', 'SibSp', 'Parch', 'Fare', 
                             'Sex_male', 'Embarked_Q', 'Embarked_S', 
                             'Pclass_2', 'Pclass_3']
            
            data = {
                'Age': [inputs_for_model['age']], 
                'SibSp': [inputs_for_model['sibsp']], 
                'Parch': [inputs_for_model['parch']], 
                'Fare': [inputs_for_model['fare']], 
                
                # One-Hot Encoding:
                'Sex_male': [1 if sex == 'male' else 0], 
                'Embarked_Q': [1 if embarked == 'Q' else 0],
                'Embarked_S': [1 if embarked == 'S' else 0],
                'Pclass_2': [1 if inputs_for_model['pclass'] == 2 else 0],
                'Pclass_3': [1 if inputs_for_model['pclass'] == 3 else 0]
            }
            
            # VIKTIG: Lager DataFrame og tvinger kolonnene til å matche FEATURE_ORDER
            input_df = pd.DataFrame(data, columns=FEATURE_ORDER)

            # --- 3. Prediksjon ---
            prediction = model.predict(input_df)[0]
            
            # Konverterer 0/1 til lesbar tekst
            survival_text = "overlevd" if prediction == 1 else "dødd"
            
            prediction_result = f"Prediksjon: Passasjeren ville ha {survival_text}."
            
        except ValueError as e:
            # Fanger feil hvis brukeren skriver inn bokstaver i tallfelt eller feil desimaltegn
            prediction_result = f"Feil i input: Sjekk at alle tall er gyldige og bruk punktum (.) for desimaler. Detalj: {e}"
        except Exception as e:
            prediction_result = f"En uventet feil oppstod under prediksjon. Detalj: {e}"

    # Dette er den kritiske endringen: Sender 'user_inputs' til HTML-malen!
    return render_template('index.html', prediction=prediction_result, inputs=user_inputs)

if __name__ == '__main__':
    app.run(debug=True)
