from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos y transformadores
model = joblib.load('modelo_random_forest.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        input_data = {
            'Name': request.form['Nombre'],
            'Sex': request.form['Sex'],
            'Fare': float(request.form['Fare']),
            'Age': float(request.form['Age']),
            'Pclass': int(request.form['Pclass']),
            'FamilySize': int(request.form['FamilySize']),
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Embarked': 'S',  # Valor por defecto (puedes cambiarlo)
            'Cabin': '1',      # Valor por defecto
            'Ticket': '12345', # Valor por defecto
        }

        # Añadir Title
        input_data['Title'] = request.form['Title']
        input_data['Name'] = input_data['Title']


        # Formatear columnas en orden correcto
        columnas_modelo = ['Name','Sex', 'Embarked', 'Cabin', 'Ticket', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Age']
        df = pd.DataFrame([input_data])
        for col in ['Embarked', 'Cabin', 'Ticket']:  # Si faltan, agregar valores ficticios
            if col not in df.columns:
                df[col] = '0'

        # Aplicar encoder ordinal a las columnas categóricas
        columnas_categoricas = ['Name','Sex', 'Embarked', 'Cabin', 'Ticket']
        df[columnas_categoricas] = encoder.transform(df[columnas_categoricas])

        # Seleccionar las columnas usadas para PCA y predicción
        df_modelo = df[['Pclass', 'Sex', 'Age', 'SibSp']]

        # Escalar y aplicar PCA
        df_scaled = scaler.transform(df_modelo)
        df_pca = pca.transform(df_scaled)

        # Predecir
        prediction = model.predict(df_pca)[0]
        resultado = 'Sobrevivió' if prediction == 1 else 'No sobrevivió'

        return jsonify({'resultado': resultado})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
