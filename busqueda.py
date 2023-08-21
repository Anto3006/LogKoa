import pandas as pd
from sklearn.model_selection import train_test_split
import json
from featureSelection import createFeatureSelectionMethod
from hyperparametros import gridSearch
from lectorParametros import LectorParametros
from modelos import crearModelo, ONLY_CPU


def busqueda(nombreArchivoDatos,nombreArchivoCrossValidation,split,porcentajeSplit):
    datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
    y_train = datos[datos.columns[1]]
    x_train = datos.drop(columns=["smiles",datos.columns[1]])
    x_train = x_train.astype("float32")

    if split:
        x_train,y_train,_,_ = train_test_split(x_train,y_train,test_size=porcentajeSplit,random_state=3006)

    archivoJson = open("parametrosGridSearch.json","r")
    datosGridSearch = json.load(archivoJson)
        

    datosHyperparametros = datosGridSearch["hyperparameters"]
    datosModelos = datosGridSearch["models"]
    datosFeatureSelection = datosGridSearch["feature_selection"]

    for nombreModelo in datosModelos:
        isUsed = datosModelos[nombreModelo]["use"]
        useGPU = (not ONLY_CPU) and datosModelos[nombreModelo]["gpu"] 
        if isUsed:
            modelo = crearModelo(nombreModelo,useGPU)
            dicModelo = datosHyperparametros[nombreModelo]
            busquedaCompleta(modelo,nombreModelo,nombreArchivoCrossValidation,x_train,y_train,dicModelo,datosFeatureSelection)


def busquedaCompleta(modelo,nombreModelo,nombreArchivoCrossValidation,x_train,y_train,diccionarioHyperparametros,datosFeatureSelection):
    for featureSelectionName in datosFeatureSelection:
        for parameters in datosFeatureSelection[featureSelectionName]:
            if parameters["use"]:
                featureSelectionMethod = createFeatureSelectionMethod(featureSelectionName,parameters)
                parameters["fileAllResults"] = nombreModelo + "_"+ featureSelectionName
                if featureSelectionName == "UFS":
                    parameters["fileAllResults"] += "_" + parameters["scoreFunc"]
                bestResult, bestHyper, bestFeatures, numberFeatures = gridSearch(modelo,featureSelectionMethod,x_train,y_train,diccionarioHyperparametros)
                guardarResultadosBusqueda(nombreArchivoCrossValidation,nombreModelo,featureSelectionName,bestResult,bestHyper,bestFeatures,numberFeatures)

def guardarResultadosBusqueda(nombreArchivo,nombreModelo,featureSelection,mejorResultado,mejoresHyper,features,numeroFeatures):
    mejoresHyper = str(mejoresHyper).replace('{','').replace('}','').replace(':','=').replace("'","").replace(' ','')
    features = str(features).replace('[','').replace(']','').replace(',','').replace("'","")
    archivo = open("CrossValidation/" + nombreArchivo,"a")
    archivo.write(nombreModelo + "," + featureSelection + "," + mejoresHyper + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + features + "\n")
    archivo.close()

def main():
    lector = LectorParametros()
    diccionarioValores = lector.leerParametros()
    nombreArchivoDatos = diccionarioValores["datos"]
    nombreArchivoCrossValidation = diccionarioValores["validacion"]
    split = diccionarioValores["split"]
    porcentajeSplit = diccionarioValores["porcentajeSplit"]
    busqueda(nombreArchivoDatos,nombreArchivoCrossValidation,split,porcentajeSplit)

if __name__ == "__main__":
    main()