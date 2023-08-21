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
    dic_hyperparametros = {}
    if len(hyperparametros) > 0:
        hyperparametros = re.sub(r'\s+', ' ', hyperparametros).split(" ")
        print(hyperparametros)
        for hyperparametro in hyperparametros:
            datosHyper = hyperparametro.split("=")
            datosHyper[0].strip()
            nombreHyper = datosHyper[0]
            valorHyper = datosHyper[1]
            isNumber = re.search("^[0-9]+.?[0-9]*$",valorHyper)
            if isNumber:
                if not "." in valorHyper:
                    valorHyper = int(valorHyper)
                else:
                    valorHyper = float(valorHyper)
            if valorHyper in ["True","False"]:
                valorHyper = bool(valorHyper)
            dic_hyperparametros[nombreHyper] = valorHyper
    return dic_hyperparametros