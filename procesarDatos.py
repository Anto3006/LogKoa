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

def columnasNa(datos,thresCol=0.2):
    columnasNa = []
    for col in datos:
        cantidadNa = datos[col].isna().sum()
        if cantidadNa > len(datos)*thresCol:
            columnasNa.append(col)
    return columnasNa

def filasNa(datos):
    filasEliminar = []
    for i in datos.index:
        if datos.loc[i].isna().sum().sum() > 0:
            filasEliminar.append(i)
    return filasEliminar


def scaleData(scaler,dataFrame):
    scaler = preprocessing.StandardScaler()
    scaler.fit(dataFrame)
    dataFrameScaled = pd.DataFrame(scaler.transform(dataFrame),columns=dataFrame.columns,index=dataFrame.index)
    return dataFrameScaled

def procesarDatos(datos,scale=False):
    y = datos[datos.columns[1]]
    x = datos.copy(deep=True)
    x.drop(columns=[datos.columns[1],"smiles"],inplace=True)
    x.replace(r'^\s*$', np.nan, regex=True,inplace=True)
    colNa = columnasNa(x)
    print(colNa)
    x.drop(columns=colNa,inplace=True)
    filNa = filasNa(x)
    print(filNa)
    x.drop(index=filNa,inplace=True)
    y.drop(index=filNa,inplace=True)
    if scale:
        scaler = preprocessing.StandardScaler()
        x = scaleData(scaler,x)
    #Eliminar datos constantes
    columnasConstantes = verificarConstancia(x)
    x.drop(columns=columnasConstantes,inplace=True)
    #Eliminar datos altamentes correlacionados
    matrizCorrelacion = x.corr().abs()
    triangularSuperior = matrizCorrelacion.where(np.triu(np.ones(matrizCorrelacion.shape),k=1).astype(np.bool_))
    colCorr = [column for column in triangularSuperior.columns if any(triangularSuperior[column] > 0.95)]
    x.drop(columns=colCorr,inplace=True)
    return x,y

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