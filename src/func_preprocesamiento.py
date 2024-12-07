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

    bin_cols_sec = ['Cisterna', 'BusinessCenter', 'Laundry', 'Jacuzzi', 'Chimenea', 'Ascensor', 'EstacionamientoVisitas', 'Lobby', 'LocalesComerciales', 'SistContraIncendios', 'PistaJogging', 'SalonFiestas', 'AreaJuegosInfantiles', 'Recepcion', 'Calefaccion', 'AccesoInternet', 'Pileta']
    bin_cols_prim = ['SalonDeUsosMul', 'AireAC', 'Estacionamiento', 'Seguridad', 'AreaParrillas', 'CanchaTennis', 'AreaCine', 'Gimnasio', 'Amoblado']
    cat_cols_prim = ['ITE_ADD_CITY_NAME', 'ITE_ADD_STATE_NAME', 'ITE_ADD_NEIGHBORHOOD_NAME']
    cat_cols_sec = ['ITE_TIPO_PROD', 'TIPOPROPIEDAD']
    num_cols = ['Dormitorios', 'Banos', 'Ambientes', 'Cocheras']
    imp_cols = ['STotalM2', 'SConstrM2', 'LONGITUDE', 'LATITUDE']
    ind_cols = num_cols + imp_cols #Columnas independientes, las que uso en RF para basarme
    #Voy agregando columnas a ind_cols a medida que las proceso

    df = preprocesar_categoricos(df, cat_cols_prim, 'frecuency', False, ind_cols)

    ind_cols = ind_cols + ['ITE_ADD_CITY_NAME_encoded', 'ITE_ADD_STATE_NAME_encoded', 'ITE_ADD_NEIGHBORHOOD_NAME_encoded']
    
    df = preprocesar_numericos(df, num_cols, 'media', ind_cols)

    df = preprocesar_binarios(df, bin_cols_prim, 'RF', ind_cols)

    ind_cols = ind_cols + bin_cols_prim

    df = preprocesar_binarios(df, bin_cols_sec, 'moda', ind_cols) #No me importan mucho

    ind_cols = ind_cols + bin_cols_sec

    df = preprocesar_numericos(df, imp_cols, 'RF', ind_cols) #STotalM2 y SConstrM2

    df = procesar_antiguedad(df, ind_cols)
    df = preprocesar_categoricos(df, cat_cols_sec, 'label', False, ind_cols)

    return df

def delete_zeros(df, columnas):
    if 'STotalM2' in columnas and 'SConstrM2' in columnas:
        df.loc[(df['STotalM2'] == 0) & (df['SConstrM2'] != 0), 'STotalM2'] = df['SConstrM2']
        df.loc[(df['SConstrM2'] == 0) & (df['STotalM2'] != 0), 'SConstrM2'] = df['STotalM2']


def preprocesar_numericos(df, columnas_numericas, imputacion='media', ind_cols=None):
    if 'STotalM2' in columnas_numericas and 'SConstrM2' in columnas_numericas:
        df.loc[(df['STotalM2'] == 0) & (df['SConstrM2'] != 0), 'STotalM2'] = df['SConstrM2']
        df.loc[(df['SConstrM2'] == 0) & (df['STotalM2'] != 0), 'SConstrM2'] = df['STotalM2']
    for columna in columnas_numericas:
        if df[columna].isnull().sum() > 0:
            if imputacion == 'media':
                media = df[columna].mean()
                df[columna] = df[columna].fillna(media).astype(int)
            if imputacion == 'RF':
                df = valor_faltante_random_forest(df, columna, 'REG', False, ind_cols)
                df[columna] = df[columna].astype(int)
    return df

def preprocesar_categoricos(df, columnas_categoricas, type_encoding='label', dupliqued=False, ind_cols=None):
    for columna in columnas_categoricas:
        if type_encoding == 'frecuency':
            frec_encoding = df[columna].value_counts() / len(df)
            df[columna + '_encoded'] = df[columna].map(frec_encoding)
            if df[columna + '_encoded'].isnull().sum() > 0:
                ind_cols = ind_cols + ['LATITUDE', 'LONGITUDE']
                df = valor_faltante_random_forest(df, columna + '_encoded', 'REG', False, ind_cols)
        elif type_encoding == 'label':
            le = LabelEncoder()
            df[columna + '_encoded'] = le.fit_transform(df[columna])

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

def procesar_antiguedad(df, ind_cols=None):
    columna = 'Antiguedad'
    df[columna] = df[columna].str.replace(' años', '')
    df[columna] = pd.to_numeric(df[columna], errors='coerce')


    df_faltantes = df[df[columna].isnull()]

    if (df_faltantes['ITE_TIPO_PROD'] == 'N').any():
        df.loc[df[columna].isnull() & (df['ITE_TIPO_PROD'] == 'N'), columna] = 0
        
    df_rf = valor_faltante_random_forest(df, columna, 'REG', False, ind_cols)

    df_rf[columna] = df_rf[columna].fillna(0).astype(int)
    
    return df_rf

def preprocesar_binarios(df, columnas_binarias, imputacion='moda', ind_cols=None):
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
            df = valor_faltante_random_forest(df, columna, 'CLAS', False, ind_cols)
        df[columna] = df[columna].astype(int)
    
    return df


def valor_faltante_random_forest(df, columna, tipo='CLAS', test=False, columnas_independientes=None):
    df_faltantes = df[df[columna].isnull()]
    df_no_faltantes = df[~df[columna].isnull()]
    print("Columna a predecir:", columna)
    if columna in columnas_independientes:
        columnas_independientes.remove(columna) 
    
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