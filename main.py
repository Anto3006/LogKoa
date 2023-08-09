import pandas as pd
from procesarDatos import procesarDatos, procesarHyperparametros
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
from guardarResultados import guardarResultadosBusqueda, evaluarModelo
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from hyperparametros import gridSearch
from featureSelection import hyperparametrosCV
from guardarResultados import generarShap
import re
import pickle
import os
import json
import sys
from lectorParametros import LectorParametros
from calcularDescriptores import calcularDescriptores
from sklearn.model_selection import train_test_split

from cuml.linear_model import LinearRegression as cumlLR
from cuml.ensemble import RandomForestRegressor as cumlRFR
from cuml.svm import LinearSVR as cumlSVR

ARCHIVO_CROSS_VALIDATION = "cross_validation_logP.csv"



def busquedaCompleta(modelo,nombreModelo,x_train,y_train,diccionarioHyperparametros,datosFeatureSelection):
    n_features = datosFeatureSelection["number_features"]

    if datosFeatureSelection["UFS"]["use"]:
        funciones_score = {"f_regression":f_regression,"r_regression":r_regression,"mutual_info_regression":mutual_info_regression}
        for funcion in funciones_score:
            nombreArchivoResultadosTotales = nombreModelo+"_"+"k_best_"+funcion
            mejorResultado, mejoresHyper, features, numeroFeatures = gridSearch(modelo,"k_best",x_train,y_train,diccionarioHyperparametros,n_features,funcion_k_best=funciones_score[funcion],nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
            guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"k_best_"+funcion,mejorResultado,mejoresHyper,features,numeroFeatures)
        
    if datosFeatureSelection["RFE"]["use"]:
        nombreArchivoResultadosTotales = nombreModelo+"_"+"rfe"
        mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"rfe",x_train,y_train,diccionarioHyperparametros,n_features,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
        guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"rfe",mejorResultado,mejoresHyper,features,numeroFeatures)

    if datosFeatureSelection["RFE_SHAP"]["use"]:    
        nombreArchivoResultadosTotales = nombreModelo+"_"+"shap"
        shap_split = datosFeatureSelection["RFE_SHAP"]["background_split"]
        mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"shap",x_train,y_train,diccionarioHyperparametros,n_features,shap_split=shap_split,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
        guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"shap",mejorResultado,mejoresHyper,features,numeroFeatures)
    
    if datosFeatureSelection["SFS"]["use"]:
        if datosFeatureSelection["SFS"]["forward"]:
            nombreArchivoResultadosTotales = nombreModelo+"_"+"sfs_forward"
            mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"sfs",x_train,y_train,diccionarioHyperparametros,n_features,direccion_sfs="forward",nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
            guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"sfs_forward",mejorResultado,mejoresHyper,features,numeroFeatures)
        if datosFeatureSelection["SFS"]["backward"]:    
            nombreArchivoResultadosTotales = nombreModelo+"_"+"sfs_backward"
            mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"sfs",x_train,y_train,diccionarioHyperparametros,n_features,direccion_sfs="backward",nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
            guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"sfs_backward",mejorResultado,mejoresHyper,features,numeroFeatures)
        

def guardarModelosMejoresCV(archivo,x_train,y_train,threshold=0.45):
    datos = archivo
    for index, row in datos.iterrows():
        if str(row["Score"]) != " " and float(row["Score"]) < threshold:
            hyp = str(row["Hyperparametros"])
            if hyp == "nan" or hyp == " ":
                hyp = ""
            nombreModelo = row["Modelo"].strip()
            print(nombreModelo)
            modelo = crearModelo(nombreModelo)
            if hyp != "":
                dicHyperparametros = procesarHyperparametros(hyp)
                modelo.set_params(**dicHyperparametros)
            features = re.sub(r'\s+', ' ', row["Features"])
            features = features.split(" ")
            features = [feature for feature in features if feature != ""]
            print(features)
            x_train_2 = x_train[features]
            modelo.fit(x_train_2,y_train)
            pickle.dump(modelo,open("Modelos/"+nombreModelo+"_"+row["Feature Selection"]+".sav",'wb'))

def evaluarModelosGuardados(x,y,nombreArchivo,tipo="test",generarFigura=True):
    modelos = os.listdir('Modelos')
    modelos.sort()
    resultados = pd.DataFrame(columns=["R2","RMSE","MUE","MSE"])
    for nombreModelo in modelos:
        modelo = pickle.load(open("Modelos/"+nombreModelo, 'rb'))
        nombreModelo = ".".join(nombreModelo.split(".")[0:-1])
        features = modelo.feature_names_in_
        x_2 = x[features]
        evaluacion = evaluarModelo(modelo,x_2,y,nombreModelo,generarFigura=generarFigura)
        resultados = resultados.append(evaluacion,ignore_index=True)
    resultados.index = modelos
    modo = "w"
    if nombreArchivo+".xlsx" in os.listdir():
        modo = "a"
    writer = pd.ExcelWriter(nombreArchivo+".xlsx",engine="openpyxl",mode=modo)
    resultados.to_excel(writer,sheet_name=tipo,engine="openpyxl")
    writer.close()
        

def crearModelo(nombreModelo,gpu=False):
    modelo = None
    if nombreModelo == "Linear Regression":
        if not gpu:
            modelo = LinearRegression()
        else:
            modelo = cumlLR(algorithm="eig")
    elif nombreModelo == "Random Forest":
        if not gpu:
            modelo = RandomForestRegressor(random_state=3006)
        else:
            modelo = cumlRFR(n_estimators=100,split_criterion=2,max_depth=5,accuracy_metric="mse",random_state=3006)
    elif nombreModelo == "SVM":
        if not gpu:
            modelo = LinearSVR(random_state=3006,max_iter=10000)
        else:
            modelo = cumlSVR(max_iter=10000,C=1.0,random_state=3006)
    elif nombreModelo == "XGBoost":
        modelo = xgb.XGBRegressor(tree_method="exact",random_state=3006)
    return modelo


def busqueda(nombreArchivo):
    datos = pd.read_csv("Datasets/"+nombreArchivo)
    y_train = datos[datos.columns[1]]
    x_train = datos.drop(columns=["smiles",datos.columns[1]])
    print(x_train)
    archivoJson = open("parametrosGridSearch.json","r")
    datosGridSearch = json.load(archivoJson)

    datosHyperparametros = datosGridSearch["hyperparameters"]
    datosModelos = datosGridSearch["models"]
    datosFeatureSelection = datosGridSearch["feature_selection"]

    for nombreModelo in datosModelos:
        if datosModelos[nombreModelo]:
            modelo = crearModelo(nombreModelo)
            dicModelo = datosHyperparametros[nombreModelo]
            busquedaCompleta(modelo,nombreModelo,x_train,y_train,dicModelo,datosFeatureSelection)

def crearDataset(datos,nombreArchivo,split,porcentajeSplit,prefijo=""):
    datos.sort_index(inplace=True)
    if split:
        datos_train,datos_test= train_test_split(datos,test_size=porcentajeSplit,random_state=3006)
        crearDataset(datos_train,nombreArchivo,split=False,porcentajeSplit=0,prefijo="train_")
        crearDataset(datos_test,nombreArchivo,split=False,porcentajeSplit=0,prefijo="test_")
    else:
        smiles = datos["smiles"].to_numpy()
        objetivo = datos[datos.columns[1]].to_numpy()
        dataset = calcularDescriptores(smiles)
        dataset.insert(1,datos.columns[1],objetivo)
        dataset.to_csv("Datasets/"+prefijo+"desc_"+nombreArchivo,index=False)

def entrenarMejoresModelos(nombreArchivoCrossValidation,nombreArchivoDatos):
    datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
    y_train = datos[datos.columns[1]]
    x_train = datos.drop(columns=["smiles",datos.columns[1]])
    archivoCV = pd.read_excel("CrossValidation/"+nombreArchivoCrossValidation)
    guardarModelosMejoresCV(archivoCV,x_train,y_train,threshold=0.6)



if __name__=="__main__":
    modo = sys.argv[1]
    lector = LectorParametros()
    diccionarioValores = lector.leerParametros()
    if modo == "desc":
        nombreArchivoDatos = diccionarioValores["datos"]
        split = diccionarioValores["split"]
        porcentajeSplit = diccionarioValores["porcentajeSplit"]
        crearDataset(pd.read_csv("Datasets/"+nombreArchivoDatos),nombreArchivoDatos,split=split,porcentajeSplit=porcentajeSplit)
    elif modo == "proc":
        nombreArchivoDatos = diccionarioValores["datos"]
        datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
        x_train,y_train = procesarDatos(datos,scale=False)
        x_train.insert(0,"smiles",datos["smiles"])
        x_train.insert(1,datos.columns[1],y_train)
        x_train.to_csv("Datasets/proc_" + nombreArchivoDatos,index=False)
    elif modo == "search":
        nombreArchivoDatos = diccionarioValores["datos"]
        busqueda(nombreArchivoDatos)
    elif modo == "train":
        nombreArchivoCrossValidation = diccionarioValores["validacion"]
        nombreArchivoDatos = diccionarioValores["datos"]
        entrenarMejoresModelos(nombreArchivoCrossValidation,nombreArchivoDatos)
    elif modo == "evaluar":
        nombreArchivoDatos = diccionarioValores["datos"]
        tipo = diccionarioValores["tipo"]
        datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
        y = datos[datos.columns[1]]
        x = datos.drop(columns=[datos.columns[1],"smiles"])
        evaluarModelosGuardados(x,y,"evaluacion_modelos",tipo=tipo,generarFigura=False)
    elif modo == "total":
        nombreArchivoDatos = diccionarioValores["datos"]
        nombreArchivoCrossValidation = diccionarioValores["validacion"]
        crearDataset(pd.read_csv("Datasets/"+nombreArchivoDatos),split=True,porcentajeSplit=porcentajeSplit)
        busqueda("train_desc_"+nombreArchivoDatos)
        entrenarMejoresModelos(nombreArchivoCrossValidation,nombreArchivoDatos)



