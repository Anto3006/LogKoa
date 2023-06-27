from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import numpy as np

def verificarConstancia(datos, threshold = 0.8):
    columnasConstantes = []
    for columna in datos:
        conteo = {}
        total = 0
        for valor in datos[columna]:
            total += 1
            if valor in conteo:
                conteo[valor] += 1
            else:
                conteo[valor] = 1
        for valor in conteo:
            conteo[valor] /= total
        for valor in conteo:
            if conteo[valor] >= threshold:
                columnasConstantes.append(columna)
    return columnasConstantes
        

def scaleData(scaler,dataFrame):
    scaler = preprocessing.StandardScaler()
    scaler.fit(dataFrame)
    dataFrameScaled = pd.DataFrame(scaler.transform(dataFrame),columns=dataFrame.columns,index=dataFrame.index)
    return dataFrameScaled

def procesarDatos(datos,scale=False):
    y = datos["log_KOA"]
    x = datos.copy(deep=True)
    x.drop(columns=["log_KOA","id", "Cas_No","smiles"],inplace=True)
    x.dropna(axis=1,how="any",inplace=True)
    if scale:
        scaler = preprocessing.StandardScaler()
        x = scaleData(scaler,x)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=3006)
    #Eliminar datos constantes
    columnasConstantes = verificarConstancia(x_train)
    x_train.drop(columns=columnasConstantes,inplace=True)
    x_test.drop(columns=columnasConstantes,inplace=True)
    #Eliminar datos altamentes correlacionados
    cor_matrix = x_train.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    x_train.drop(columns=to_drop,inplace=True)
    x_test.drop(columns=to_drop,inplace=True)
    return x_train,y_train,x_test,y_test

def procesarHyperparametros(hyperparametros):
    hyperparametros = re.sub(r'\s+', ' ', hyperparametros).split(" ")
    dic_hyperparametros = {}
    for hyperparametro in hyperparametros:
        print(hyperparametro)
        datosHyper = hyperparametro.split("=")
        datosHyper[0].strip()
        nombreHyper = datosHyper[0]
        valorHyper = float(datosHyper[1])
        if nombreHyper in ["max_depth","n_estimators"]:
            valorHyper = int(valorHyper)
        dic_hyperparametros[nombreHyper] = valorHyper
    return dic_hyperparametros