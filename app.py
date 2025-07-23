from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelos
rules_df = joblib.load('rules.pkl')
df_transacciones = pd.read_csv('atelier-datshet-2024-transacciones.csv')  # Asegúrate que sea el mismo nombre exacto

# Función de recomendación
def recomendar_por_similitud(atributos_elegidos, rules_df, df, n=3):
    if not rules_df.empty:
        mejores_reglas = []
        for _, row in rules_df.iterrows():
            antecedent = set(row['antecedents'])
            coincidencias = len(antecedent.intersection(atributos_elegidos))
            if coincidencias > 0:
                mejores_reglas.append((row, coincidencias))

        if mejores_reglas:
            mejores_reglas.sort(key=lambda x: (x[1], x[0]['lift'], x[0]['confidence']), reverse=True)
            mejor_regla = mejores_reglas[0][0]
            recomendaciones = list(set(mejor_regla['consequents']) - atributos_elegidos)
            return recomendaciones[:n], 'regla_mas_similar'

    # Fallback con atributos más frecuentes
    columnas = ['producto_talla', 'producto_color', 'producto_temporada', 'vestido_estilo', 'vestido_condicion']
    valores = []
    for col in columnas:
        if col in df.columns:
            valores.extend(df[col].dropna().apply(lambda x: f"{col}_{x}"))
    populares = pd.Series(valores).value_counts().index.tolist()
    recomendaciones = [a for a in populares if a not in atributos_elegidos][:n]
    return recomendaciones, 'atributos_populares'

# Endpoint de recomendación
@app.route('/api/recomendar-atributos', methods=['POST'])
def recomendar():
    datos = request.json
    try:
        atributos = frozenset([
            f"producto_talla_{datos['producto_talla']}",
            f"producto_color_{datos['producto_color']}",
            f"producto_temporada_{datos['producto_temporada']}",
            f"vestido_estilo_{datos['vestido_estilo']}",
            f"vestido_condicion_{datos['vestido_condicion']}"
        ])
    except KeyError as e:
        return jsonify({'error': f'Falta el atributo: {str(e)}'}), 400

    recomendaciones, metodo = recomendar_por_similitud(atributos, rules_df, df_transacciones)

    return jsonify({
        'input_atributos': list(atributos),
        'recomendaciones': recomendaciones,
        'metodo': metodo
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
