import pandas as pd
from sklearn.model_selection import train_test_split
import json
from featureSelection import createFeatureSelectionMethod
from hyperparametros import gridSearch
from lectorParametros import LectorParametros
from modelos import crearModelo, ONLY_CPU
import os


def busqueda(nombreArchivoDatos,nombreArchivoCrossValidation,split,porcentajeSplit):
    datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
    y_train = datos[datos.columns[1]]
    x_train = datos.drop(columns=["smiles",datos.columns[1]])
    x_train = x_train.astype("float32")

    if split:
        x_train,_,y_train,_ = train_test_split(x_train,y_train,test_size=porcentajeSplit,random_state=3006)

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
            busquedaCompleta(modelo,useGPU,nombreModelo,nombreArchivoCrossValidation,x_train,y_train,dicModelo,datosFeatureSelection)


def busquedaCompleta(modelo,useGPU,nombreModelo,nombreArchivoCrossValidation,x_train,y_train,diccionarioHyperparametros,datosFeatureSelection):
    for featureSelectionName in datosFeatureSelection:
        for parameters in datosFeatureSelection[featureSelectionName]:
            if parameters["use"]:
                featureSelectionMethod = createFeatureSelectionMethod(featureSelectionName,parameters)
                parameters["fileAllResults"] = nombreModelo + "_"+ featureSelectionName
                if featureSelectionName == "UFS":
                    parameters["fileAllResults"] += "_" + parameters["scoreFunc"]
                elif featureSelectionName == "SFS":
                    parameters["fileAllResults"] += "_" + parameters["direction"]
                bestResult, bestHyper, bestFeatures, numberFeatures = gridSearch(modelo,featureSelectionMethod,x_train,y_train,diccionarioHyperparametros)
                guardarResultadosBusqueda(nombreArchivoCrossValidation,nombreModelo,useGPU,featureSelectionName,parameters,bestResult,bestHyper,bestFeatures,numberFeatures)

def guardarResultadosBusqueda(nombreArchivo,nombreModelo,useGPU,featureSelection,parametrosFeatureSelection,mejorResultado,mejoresHyper,features,numeroFeatures):
    parametrosFeatureSelection = [(parametro,parametrosFeatureSelection[parametro]) for parametro in parametrosFeatureSelection if parametro not in ["use","number_features","fileAllResults"]]
    parametrosFeatureSelection = str(parametrosFeatureSelection).replace('[','').replace(']','').replace(',','').replace("'","")
    mejoresHyper = str(mejoresHyper).replace('{','').replace('}','').replace(':','=').replace("'","").replace(' ','')
    features = str(features).replace('[','').replace(']','').replace(',','').replace("'","")
    if not os.path.exists("CrossValidation/" + nombreArchivo):
        archivo = open("CrossValidation/" + nombreArchivo,"w")
        archivo.write("Model,GPU,FS,param_FS,Hyperparameters,Score,Number Features,Features\n")
    else:
        archivo = open("CrossValidation/" + nombreArchivo,"a")
    archivo.write(nombreModelo + "," + str(useGPU) + "," + featureSelection + "," + parametrosFeatureSelection + "," + mejoresHyper + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + features + "\n")
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