import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# num_cols = ['Dormitorios', 'Banos', 'Ambientes', 'Cocheras','Amoblado','Antiguedad','ITE_TIPO_PROD_encoded','Laundry','Calefaccion','Jacuzzi','Gimnasio','Cisterna','AireAC','SalonFiestas']
# imp_cols = ['STotalM2', 'SConstrM2', 'LONGITUDE', 'LATITUDE']
def preprocesar(df, tipo='train', extra_cols=None):
    """
    Preprocesa el dataframe de acuerdo a las columnas que se consideran importantes.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame con los datos a preprocesar.
    tipo : str, optional
        Tipo de preprocesamiento a realizar. Por defecto es 'train', en este caso se eliminan muestras, sino no.
    extra_cols : list, optional
        Lista con columnas adicionales a considerar en el preprocesamiento, pero estas no se procesan. Por defecto es None.
    """
    imp_cols = [
    'STotalM2', 'SConstrM2', 'LONGITUDE', 'LATITUDE', 'Dormitorios', 'Banos', 
    'Ambientes', 'Cocheras', 'Amoblado', 'Antiguedad', 'ITE_TIPO_PROD', 
    'Laundry', 'Calefaccion', 'Jacuzzi', 'Gimnasio', 'Cisterna', 'AireAC', 'SalonFiestas', 
    'precio_pesos_constantes'
    ]
    if extra_cols is not None:
        imp_cols = imp_cols + extra_cols

    df = delete_columns(df, imp_cols)
    
    binarias = ['Laundry', 'Calefaccion', 'Jacuzzi', 'Gimnasio', 'Cisterna', 'AireAC', 'SalonFiestas', 'Amoblado']
    categoricas = ['ITE_TIPO_PROD']
    numericas = ['Dormitorios', 'Banos', 'Ambientes', 'Cocheras']

    df = preprocesar_categoricos(df, categoricas, 'label')
    df = preprocesar_binarios(df, binarias, 'RF')
    df = preprocesar_numericos(df, numericas, 'RF') 
    df = procesar_antiguedad(df)
    df = acotar_caracteristicas(df, tipo)
    df = df.drop(columns=['ITE_TIPO_PROD'])

    return df

def delete_zeros(df, columnas):
    if 'STotalM2' in columnas and 'SConstrM2' in columnas:
        df.loc[(df['STotalM2'] == 0) & (df['SConstrM2'] != 0), 'STotalM2'] = df['SConstrM2']
        df.loc[(df['SConstrM2'] == 0) & (df['STotalM2'] != 0), 'SConstrM2'] = df['STotalM2']

def acotar_caracteristicas(df, tipo='train'):
    #Columnas que no se aceptan valores faltantes
    columnas_faltantes = ['STotalM2', 'SConstrM2', 'LONGITUDE', 'LATITUDE']
    if tipo == 'train':
        df = df.dropna(subset=columnas_faltantes)
    else:
        for columna in columnas_faltantes:
            df = valor_faltante_random_forest(df, columna, 'REG', False)
    
    #Poner STotalM2 y SConstrM2 enteros
    df.loc[:, 'STotalM2'] = df['STotalM2'].astype(int)
    df.loc[:, 'SConstrM2'] = df['SConstrM2'].astype(int)

    #Si alguno de los dos es 0, se reemplaza por el valor del otro
    df.loc[(df['STotalM2'] == 0) & (df['SConstrM2'] != 0), 'STotalM2'] = df['SConstrM2']
    df.loc[(df['SConstrM2'] == 0) & (df['STotalM2'] != 0), 'SConstrM2'] = df['STotalM2']

    if tipo == 'train':
        #Metros cuadrados
        df = df[(df['STotalM2'] > 10) & (df['STotalM2'] < 10**3)]
        df = df[(df['SConstrM2'] > 10) & (df['SConstrM2'] < 10**3)]

        #Dormitorios
        df = df[(df['Dormitorios'] >= 0) & (df['Dormitorios'] < 10)]

        #Banos
        df = df[(df['Banos'] > 0) & (df['Banos'] < 10)]

        #Ambientes
        df = df[(df['Ambientes'] > 0) & (df['Ambientes'] < 20)]

        #Cocheras
        df = df[(df['Cocheras'] >= 0) & (df['Cocheras'] < 10)]
    else:
        #Metros cuadrados
        df.loc[df['STotalM2'] <= 10, 'STotalM2'] = 11
        df.loc[df['STotalM2'] >= 10**3, 'STotalM2'] = 999
        df.loc[df['SConstrM2'] <= 10, 'SConstrM2'] = 11
        df.loc[df['SConstrM2'] >= 10**3, 'SConstrM2'] = 999

        # Dormitorios
        df.loc[df['Dormitorios'] < 0, 'Dormitorios'] = 0
        df.loc[df['Dormitorios'] >= 10, 'Dormitorios'] = 9

        # Banos
        df.loc[df['Banos'] <= 0, 'Banos'] = 1
        df.loc[df['Banos'] >= 10, 'Banos'] = 9

        # Ambientes
        df.loc[df['Ambientes'] <= 0, 'Ambientes'] = 1
        df.loc[df['Ambientes'] >= 20, 'Ambientes'] = 19

        # Cocheras
        df.loc[df['Cocheras'] < 0, 'Cocheras'] = 0
        df.loc[df['Cocheras'] >= 10, 'Cocheras'] = 9

    return df


def preprocesar_numericos(df, columnas_numericas, imputacion='media', ind_cols=None):
    for columna in columnas_numericas:
        if df[columna].isnull().sum() > 0:
            if imputacion == 'media':
                media = df[columna].mean()
                df[columna] = df[columna].fillna(media).astype(int)
            if imputacion == 'RF':
                ind_cols = df.drop(columns=['Antiguedad']).columns.tolist()
                print("Las cols independientes son:", ind_cols)
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

def delete_columns(df, imp_cols):
    #Elimina las columnas que no son importantes
    del_cols = [col for col in df.columns if col not in imp_cols]
    return df.drop(columns=del_cols)

def procesar_antiguedad(df):
    columna = 'Antiguedad'
    df[columna] = df[columna].str.replace(' años', '')
    df[columna] = pd.to_numeric(df[columna], errors='coerce')

    df_faltantes = df[df[columna].isnull()]

    if (df_faltantes['ITE_TIPO_PROD'] == 'N').any():
        df.loc[df[columna].isnull() & (df['ITE_TIPO_PROD'] == 'N'), columna] = 0
        
    df_rf = valor_faltante_random_forest(df, columna, 'REG', False)
    df_rf[columna] = df_rf[columna].astype(int)

    df_rf = df_rf[(df_rf['Antiguedad'] >= 0) & (df_rf['Antiguedad'] < 150)]
    
    return df_rf

def normalizar_valor(valor):
        if pd.isnull(valor):
            return np.nan
        valor = str(valor).strip().lower()
        valor = re.sub(r'\s+', '', valor)
        return valor

def preprocesar_binarios(df, columnas_binarias, imputacion='moda'):
    mapeo_binario = {
        '0': 0, 'no': 0, '0.0': 0,
        '1': 1, 'si': 1, '1.0': 1, 'sí': 1
    }
    
    for columna in columnas_binarias:
        df[columna] = df[columna].map(lambda x: mapeo_binario.get(normalizar_valor(x), x))

        if imputacion == 'moda':
            moda = df[columna].mode()[0]
            df[columna] = df[columna].fillna(moda)
        if imputacion == 'media':
            mediana = df[columna].median()
            df[columna] = df[columna].fillna(mediana)
        if imputacion == 'RF':
            ind_cols = ['STotalM2', 'SConstrM2', 'LONGITUDE', 'LATITUDE']
            df = valor_faltante_random_forest(df, columna, 'CLAS', False, ind_cols)
        df[columna] = df[columna].astype(int)
    
    return df


def valor_faltante_random_forest(df, columna, tipo='CLAS', test=False, ind_cols=None):
    if df[columna].isnull().sum() == 0:
        return df
    df_faltantes = df[df[columna].isnull()]
    df_no_faltantes = df[~df[columna].isnull()]
    print("Columna a predecir:", columna)
    if ind_cols is None:
        #Eliminar columna del dataframe
        columnas_independientes = df.drop(columns=[columna, 'ITE_TIPO_PROD', 'precio_pesos_constantes']).columns.tolist()
    else:
        columnas_independientes = ind_cols

    
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