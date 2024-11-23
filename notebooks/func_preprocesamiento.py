import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocesar(df):
    del_cols = ['id_grid', 'MesListing', 'SitioOrigen', 'year']
    df = delete_columns(df, del_cols)

    bin_cols_sec = ['Cisterna', 'BusinessCenter', 'Laundry', 'Jacuzzi', 'Chimenea', 'Ascensor', 'EstacionamientoVisitas', 'Lobby', 'LocalesComerciales', 'SistContraIncendios', 'PistaJogging', 'SalonFiestas', 'AreaJuegosInfantiles', 'Recepcion', 'Calefaccion']
    bin_cols_prim = ['SalonDeUsosMul', 'AireAC', 'Estacionamiento', 'Seguridad', 'AreaParrilla', 'CanchaTenis', 'AreaCine', 'Gimnasio']
    cat_cols_prim = ['ITE_ADD_CITY_NAME', 'ITE_ADD_STATE_NAME', 'ITE_ADD_NEIGHBORHOOD_NAME']
    cat_cols_sec = ['ITE_TIPO_PROD', 'TIPOPROPIEDAD']
    num_cols = ['Dormitorios', 'Banos', 'Ambientes', 'Cocheras']
    imp_cols = ['Antiguedad', 'STotalM2', 'SConstrM2']

    df = preprocesar_numericos(df, num_cols, 'RF')


def preprocesar_numericos(df, columnas_numericas, imputacion='media'):
    for columna in columnas_numericas:
        if imputacion == 'media':
            media = df[columna].mean()
            df[columna] = df[columna].fillna(media)
        if imputacion == 'RF':
            df = valor_faltante_random_forest(df, columna)
    
    return df

def preprocesar_categoricos(df, columnas_categoricas, type_encoding='label', dupliqued=False):
    for columna in columnas_categoricas:
        if type_encoding == 'frecuency':
            frec_encoding = df[columna].value_counts() / len(df)
            df[columna + '_encoded'] = df[columna].map(frec_encoding)
            if df[columna + '_encoded'].isnull().sum() > 0:
                df = valor_faltante_random_forest(df, columna + '_encoded', 'REG')
        elif type_encoding == 'label':
            le = LabelEncoder()
            df[columna + '_encoded'] = le.fit_transform(df[columna])
            print(df[[columna, columna + '_encoded']])

    if dupliqued: #Se puede borrar a futuro, es solo chequeo
        for columna in columnas_categoricas:
            # Agrupar por el valor codificado y contar los barrios únicos
            duplicados = df.groupby(columna + '_encoded')[columna].nunique()
            duplicados = duplicados[duplicados > 1]
            
            if not duplicados.empty:
                print(f'Hay duplicados en la columna {columna + "_encoded"}')
                # Imprimir las ciudades que tienen el mismo valor de frecuencia
                for valor in duplicados.index:
                    ciudades = df[df[columna + '_encoded'] == valor][columna].unique()
                    print(f'Valor codificado: {valor}, Ciudades: {ciudades}')
        
    return df



def delete_columns(df, columnas):
    return df.drop(columnas, axis=1)

def procesar_antiguedad(df):
    columna = 'Antiguedad'
    df[columna] = df[columna].str.replace(' años', '')
    df[columna] = pd.to_numeric(df[columna], errors='coerce')


    df_faltantes = df[df[columna].isnull()]

    if (df_faltantes['ITE_TIPO_PROD'] == 'N').any():
        df.loc[df[columna].isnull() & (df['ITE_TIPO_PROD'] == 'N'), columna] = 0
        
    df_rf = valor_faltante_random_forest(df, columna)

    df_rf[columna] = df_rf[columna].fillna(0).astype(int)
    
    return df_rf

def preprocesar_binarios(df, columnas_binarias, imputacion='moda'):
    mapeo_binario = {
        '0': 0, 'no': 0, '0.0': 0,
        '1': 1, 'si': 1, '1.0': 1, 'sí': 1
    }
    
    def normalizar_valor(valor):
        if pd.isnull(valor):
            return np.nan
        valor = str(valor).strip().lower()
        valor = re.sub(r'\s+', '', valor)
        return valor
    
    for columna in columnas_binarias:
        df[columna] = df[columna].map(lambda x: mapeo_binario.get(normalizar_valor(x), x))

        if imputacion == 'moda':
            moda = df[columna].mode()[0]
            df[columna] = df[columna].fillna(moda)
        if imputacion == 'media':
            mediana = df[columna].median()
            df[columna] = df[columna].fillna(mediana)
        if imputacion == 'RF':
            df = valor_faltante_random_forest(df, columna)
        df[columna] = df[columna].astype(int)
    
    return df


def valor_faltante_random_forest(df, columna, tipo='CLAS', test=False):
    df_faltantes = df[df[columna].isnull()]
    df_no_faltantes = df[~df[columna].isnull()]
    columnas_independientes = ['STotalM2', 'SConstrM2', 'Dormitorios', 'Banos', 'Ambientes']  # Todas las columnas menos la columna objetivo
    
    df_no_faltantes = df_no_faltantes.dropna(subset=columnas_independientes)

    X_faltantes = df_faltantes[columnas_independientes]
    X = df_no_faltantes[columnas_independientes]
    y = df_no_faltantes[columna]
    

    if tipo == 'CLAS':
        modelo = RandomForestClassifier(n_estimators=100, random_state=46)
    elif tipo == 'REG':
        modelo = RandomForestRegressor(n_estimators=100, random_state=46)

    if test:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelo.fit(X_train, y_train)
        y_val_pred = modelo.predict(X_val)
    
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        print("Vector real:", y_val.values)
        print("Vector predicho:", y_val_pred)
        print("RMSE:", rmse)
    else:
        modelo.fit(X, y)
    
    y_faltantes_pred = modelo.predict(X_faltantes)
    df.loc[df[columna].isnull(), columna] = y_faltantes_pred
    
    return df