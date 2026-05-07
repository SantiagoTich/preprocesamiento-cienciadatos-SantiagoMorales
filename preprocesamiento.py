import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocesar_datos(df):
    print("Iniciando preprocesamiento completo...")

    # 1. Eliminación de duplicados
    df_limpio = df.drop_duplicates().copy()
    print("✓ Duplicados eliminados.")

    # 2. Gestión de valores nulos
    for col in df_limpio.columns:
        if df_limpio[col].dtype == 'object':
            df_limpio[col] = df_limpio[col].fillna('Desconocido')
        else:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mean())
    print("✓ Valores nulos gestionados (Media para numéricos, 'Desconocido' para categóricos).")

    # 3. Codificación de variables categóricas
    for col in df_limpio.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_limpio[col] = le.fit_transform(df_limpio[col])
    print("✓ Variables categóricas codificadas (Label Encoding).")

    # 4. Normalización
    scaler = MinMaxScaler()
    columnas_numericas = df_limpio.select_dtypes(include=['float64', 'int64', 'int32']).columns
    df_limpio[columnas_numericas] = scaler.fit_transform(df_limpio[columnas_numericas])
    print("✓ Datos numéricos normalizados (Min-Max Scaler).")

    return df_limpio

# --- EJEMPLO DE PRUEBA ---
if __name__ == "__main__":
    # Creamos un dataset de prueba para verificar que las funciones sirven
    data = {
        'ID': [1, 2, 2, 4, 5], # Contiene un duplicado (fila 2 y 3)
        'Edad': [20, 25, 25, None, 22], # Contiene un nulo
        'Ciudad': ['Riobamba', 'Quito', 'Quito', 'Guayaquil', None], # Contiene nulos y texto
        'Gasto_Mensual': [150.5, 200.0, 200.0, 350.75, None]
    }
    df_prueba = pd.DataFrame(data)
    
    print("--- DATASET ORIGINAL ---")
    print(df_prueba, "\n")
    
    df_resultado = preprocesar_datos(df_prueba)
    
    print("\n--- DATASET PREPROCESADO ---")
    print(df_resultado)
