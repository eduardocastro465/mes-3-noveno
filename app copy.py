from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json

app = Flask(__name__)

# --- Carga de Modelos y Datos (sin cambios, es lo mejor) ---
RULES_FILE = 'rules_recomendacion.json'
DATA_FILE = 'atelier-datshet-2024-transacciones.csv'

rules_df = None
df_completo = None

def cargar_modelos_y_datos():
    """Carga los modelos de reglas y el DataFrame de datos."""
    global rules_df, df_completo
    try:
        if os.path.exists(RULES_FILE):
            rules_df = pd.read_json(RULES_FILE)
            rules_df['antecedents'] = rules_df['antecedents'].apply(frozenset)
            rules_df['consequents'] = rules_df['consequents'].apply(frozenset)
            print("Reglas de asociación cargadas exitosamente.")
        else:
            print("Archivo de reglas no encontrado. Asegúrate de que rules_recomendacion.json exista.")
            rules_df = pd.DataFrame()
            
        if os.path.exists(DATA_FILE):
            df_completo = pd.read_csv(DATA_FILE)
            if 'producto_id' not in df_completo.columns:
                df_completo['producto_id'] = range(1, len(df_completo) + 1)
            print("DataFrame de datos cargado exitosamente.")
        else:
            print(f"Advertencia: El archivo de datos '{DATA_FILE}' no se encontró. Las recomendaciones serán limitadas.")
            df_completo = pd.DataFrame()

    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        rules_df = pd.DataFrame()
        df_completo = pd.DataFrame()

cargar_modelos_y_datos()

# --- Funciones de Recomendación (sin cambios) ---

def recomendar_por_similitud(atributos_elegidos, rules_df, df_transacciones_para_fallback, num_recomendaciones=2):
    # (El código de esta función es el mismo que en la respuesta anterior)
    if not rules_df.empty:
        mejores_reglas = []
        for _, row in rules_df.iterrows():
            antecedent = row['antecedents']
            coincidencias = len(antecedent.intersection(atributos_elegidos))
            if coincidencias > 0:
                mejores_reglas.append({
                    'regla': row,
                    'coincidencias': coincidencias
                })
        
        if mejores_reglas:
            mejores_reglas.sort(key=lambda x: (x['coincidencias'], x['regla']['lift'], x['regla']['confidence']), reverse=True)
            mejor_regla = mejores_reglas[0]['regla']
            recomendaciones = list(mejor_regla['consequents'])
            atributos_finales = [item for item in recomendaciones if item not in atributos_elegidos]
            return atributos_finales[:num_recomendaciones], "regla_mas_similar"

    if df_transacciones_para_fallback.empty:
        return [], "no_hay_datos"
        
    columnas_atributos = ['producto_talla', 'producto_color', 'producto_temporada', 'vestido_estilo', 'vestido_condicion']
    list_of_attributes = [df_transacciones_para_fallback[attr].apply(lambda x: f"{attr}_{x}") 
                          for attr in columnas_atributos if attr in df_transacciones_para_fallback.columns and not df_transacciones_para_fallback[attr].empty]
    if not list_of_attributes:
        return [], "no_hay_datos"
    atributos_populares_serie = pd.concat(list_of_attributes)
    atributos_populares = (atributos_populares_serie.value_counts().head(num_recomendaciones + len(atributos_elegidos)).index.tolist())
    general_popular_attributes = [attr for attr in atributos_populares if attr not in atributos_elegidos]
    return general_popular_attributes[:num_recomendaciones], "atributos_populares"

def encontrar_vestidos_por_atributos(df_original, atributos_recomendados, excluidos=None):
    # (El código de esta función es el mismo que en la respuesta anterior)
    if not atributos_recomendados:
        return pd.DataFrame()
    filtros = []
    columnas_mapeo = {
        'producto_talla': 'producto_talla',
        'producto_color': 'producto_color',
        'producto_temporada': 'producto_temporada',
        'vestido_estilo': 'vestido_estilo',
        'vestido_condicion': 'vestido_condicion'
    }
    for attr_str in atributos_recomendados:
        for prefix, col_name in columnas_mapeo.items():
            if attr_str.startswith(f"{prefix}_"):
                valor = attr_str.replace(f"{prefix}_", "", 1)
                filtros.append(df_original[col_name] == valor)
                break
    
    if not filtros:
        return pd.DataFrame()
    
    filtro_combinado = filtros[0]
    for i in range(1, len(filtros)):
        filtro_combinado &= filtros[i]
        
    resultados = df_original[filtro_combinado].copy()
    if excluidos:
        resultados = resultados[~resultados['producto_nombre'].isin(excluidos)]
        
    return resultados.head(5)

# --- Rutas de Flask ---

@app.route('/recomendar', methods=['POST'])
def recomendar_productos():
    """
    Recibe un JSON con los atributos del producto y devuelve una lista de nombres de productos.
    """
    try:
        # 1. Obtener los datos del JSON
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "mensaje": "Se esperaba un cuerpo JSON."}), 400

        required_fields = ['producto_nombre', 'producto_talla', 'producto_color', 'producto_temporada', 'vestido_estilo', 'vestido_condicion']
        if not all(field in data for field in required_fields):
            return jsonify({"success": False, "mensaje": "Faltan campos requeridos en el JSON."}), 400

        datos_transaccion = data
        
        # 2. Formatear los atributos para el modelo
        atributos_elegidos = frozenset([
            f"producto_talla_{datos_transaccion['producto_talla']}",
            f"producto_color_{datos_transaccion['producto_color']}",
            f"producto_temporada_{datos_transaccion['producto_temporada']}",
            f"vestido_estilo_{datos_transaccion['vestido_estilo']}",
            f"vestido_condicion_{datos_transaccion['vestido_condicion']}"
        ])
        
        # 3. Hacer la recomendación de atributos
        atributos_recomendados, metodo = recomendar_por_similitud(
            atributos_elegidos=atributos_elegidos,
            rules_df=rules_df,
            df_transacciones_para_fallback=df_completo,
            num_recomendaciones=3
        )
        
        # 4. Buscar los productos que coinciden
        vestidos_recomendados = encontrar_vestidos_por_atributos(
            df_original=df_completo,
            atributos_recomendados=atributos_recomendados,
            excluidos=[datos_transaccion['producto_nombre']]
        )
        
        # 5. Preparar la respuesta JSON con los nombres de los productos
        if not vestidos_recomendados.empty:
            recomendaciones_lista = vestidos_recomendados['producto_nombre'].tolist()
            mensaje = "¡Aquí tienes algunas recomendaciones basadas en tu compra!"
        else:
            recomendaciones_lista = []
            mensaje = "Lo siento, no encontramos productos que coincidan con la recomendación."

        return jsonify({
            "success": True,
            "mensaje": mensaje,
            "metodo_usado": metodo,
            "productos_recomendados": recomendaciones_lista
        })

    except Exception as e:
        print(f"Error en la ruta de recomendación: {e}")
        return jsonify({
            "success": False,
            "mensaje": f"Ocurrió un error: {e}",
            "productos_recomendados": []
        }), 500

if __name__ == '__main__':
    # Para producción, se recomienda no usar debug=True
    app.run(debug=True)