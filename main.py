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


ARCHIVO_CROSS_VALIDATION = "cross_validation.csv"



def busquedaCompleta(modelo,nombreModelo,x_train,y_train,diccionarioHyperparametros,n_features="best"):
    funciones_score = {"f_regression":f_regression,"r_regression":r_regression,"mutual_info_regression":mutual_info_regression}
    for funcion in funciones_score:
        nombreArchivoResultadosTotales = nombreModelo+"_"+"k_best_"+funcion
        mejorResultado, mejoresHyper, features, numeroFeatures = gridSearch(modelo,"k_best",x_train,y_train,diccionarioHyperparametros,n_features,funcion_k_best=funciones_score[funcion],nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
        guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"k_best_"+funcion,mejorResultado,mejoresHyper,features,numeroFeatures)
    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"rfe"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"rfe",x_train,y_train,diccionarioHyperparametros,n_features,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"rfe",mejorResultado,mejoresHyper,features,numeroFeatures)

    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"shap"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"shap",x_train,y_train,diccionarioHyperparametros,n_features,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"shap",mejorResultado,mejoresHyper,features,numeroFeatures)
    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"sfs_forward"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"sfs",x_train,y_train,diccionarioHyperparametros,n_features,direccion_sfs="forward",nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"sfs_forward",mejorResultado,mejoresHyper,features,numeroFeatures)
    

    
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
            modelo = obtenerModelo(nombreModelo,hyp)
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
        

def obtenerModelo(nombreModelo,hyperparametros):
    modelo = None
    if nombreModelo == "Linear Regression":
        modelo = LinearRegression()
    elif nombreModelo == "Random Forest":
        modelo = RandomForestRegressor(random_state=3006)
    elif nombreModelo == "SVM":
        modelo = LinearSVR(random_state=3006,max_iter=10000)
    elif nombreModelo == "XGBoost":
        modelo = xgb.XGBRegressor(tree_method="exact",random_state=3006)
    
    if hyperparametros != "":
        dic_hyperparametros = procesarHyperparametros(hyperparametros)
        modelo.set_params(**dic_hyperparametros)
    return modelo


if __name__=="__main__":
    modo = sys.argv[1]
    lector = LectorParametros()
    diccionarioValores = lector.leerParametros()
    if modo == "search":
        nombreArchivo = diccionarioValores["datos"]
        datos = pd.read_csv("Datasets/"+nombreArchivo)
        x_train,y_train,x_test,y_test = procesarDatos(datos,scale=False)
        archivoJson = open("parametrosGridSearch.json","r")
        datosGridSearch = json.load(archivoJson)

        dic_xgboost = datosGridSearch["xgboost"]
        dic_forest = datosGridSearch["random_forest"]
        dic_SVM = datosGridSearch["SVM"]
        modelo = LinearRegression()
        nombreModelo = "Linear Regression"
        busquedaCompleta(modelo,nombreModelo,x_train,y_train,{},n_features="best")


        modelo = RandomForestRegressor(random_state=3006)
        nombreModelo = "Random Forest"
        busquedaCompleta(modelo,nombreModelo,x_train,y_train,dic_forest,n_features="best")


        modelo = xgb.XGBRegressor(tree_method="exact",random_state=3006)
        nombreModelo = "XGBoost"
        busquedaCompleta(modelo,nombreModelo,x_train,y_train,dic_xgboost,n_features="best")

        modelo = LinearSVR(random_state=3006,max_iter=10000)
        nombreModelo = "SVM"
        busquedaCompleta(modelo,nombreModelo,x_train,y_train,dic_SVM,n_features="best")
        guardarModelosMejoresCV(archivo,x_train,y_train,threshold=0.45)
        pass
    elif modo == "train":
        nombreArchivoCrossValidation = diccionarioValores["validacion"]
        nombreArchivoDatos = diccionarioValores["datos"]
        datos = pd.read_csv("Datasets/"+nombreArchivo)
        x_train,y_train,_,_ = procesarDatos(datos,scale=False)
        archivo = pd.read_excel("CrossValidation/"+nombreArchivo)
        guardarModelosMejoresCV(archivo,x_train,y_train,threshold=0.6)
        pass
    elif modo == "evaluar":
        nombreArchivoDatos = diccionarioValores["datos"]
        tipo = diccionarioValores["tipo"]
        datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
        y = datos["log_Koa"]
        x = datos.drop(columns=["log_Koa"])
        evaluarModelosGuardados(x,y,"evaluacion_modelos",tipo=tipo,generarFigura=False)
        pass


datos = pd.read_csv("Datasets/logKoa_descriptors_total.csv")

x_train,y_train,x_test,y_test = procesarDatos(datos,scale=False)

#x_train_scaled,_,x_test_scaled,_ = procesarDatos(datos,scale=True)

datosTest2 = pd.read_csv("Datasets/externalCanon_descriptors_total_final_2.csv")
datosTest2.index = datosTest2["No."]
y_external = datosTest2["log_Koa"]
x_external = datosTest2.drop(columns=["No.","Compound.name","log_Koa","smiles","CAS.RN"])
x_external = x_external[x_train.columns]

x_test_external = pd.concat([x_test,x_external],ignore_index=True,axis=0)
y_test_external = pd.concat([y_test,y_external])

x_total = pd.concat([x_train,x_test,x_external],ignore_index=True,axis=0)
y_total = pd.concat([y_train,y_test,y_external],ignore_index=True,axis=0)




#archivo = pd.read_excel("CrossValidation/cross_validation_no_limit.xlsx")

#guardarModelosMejoresCV(archivo,x_train,y_train,threshold=0.45)

#evaluarModelosGuardados(x_total,y_total,"evaluacion_modelos",tipo="total",generarFigura=False)

